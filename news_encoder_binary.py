# %%
import warnings
warnings.filterwarnings('ignore')

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import pandas as pd

import re, json, requests

from datetime import date, datetime, timedelta

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import IncrementalPCA

from transformers import BertForSequenceClassification, AutoTokenizer

# %%
import os
import torch
import random
import numpy as np

def seed_everything(seed):
    global SEED
    SEED = seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

dtype = torch.bfloat16
torch.set_default_dtype(dtype)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
candles = pd.concat([pd.read_csv('data/candles.csv'), pd.read_csv('data/candles_2.csv')])
candles['begin'] = pd.to_datetime(candles['begin'])
candles = candles.drop_duplicates().sort_values(by=['ticker', 'begin']).reset_index(drop=True)

news = pd.concat([pd.read_csv('data/news.csv'), pd.read_csv('data/news_2.csv')]).drop(columns=['Unnamed: 0'])
news['publish_date'] = pd.to_datetime(news['publish_date'])
news = news.drop_duplicates().sort_values(by=['publish_date', 'title']).reset_index(drop=True)

# %%
last_date = candles['begin'].max()

# %%
candles['train_target'] = candles.groupby('ticker')['close'].transform(lambda x: x.shift(-14) / x - 1)
candles = candles.dropna().reset_index(drop=True)

# %%
model_path = 'unthinkable/RuFinBERT_turbo_multitarget'
tokenizer = AutoTokenizer.from_pretrained(model_path)
finbert = BertForSequenceClassification.from_pretrained(model_path).to(device, dtype)

# %%
hidden_dim1 = finbert.classifier.in_features
hidden_dim2 = 512

# %%
class Connector(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.connector = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_out)
        )

    def forward(self, x):
        return self.connector(x)

# %%
class SinusoidalTimeEncoding(nn.Module):
    """
    Синусоидальное кодирование времени (давности) как в Vaswani et al.
    На вход подаются возраста (в днях) размера (B, N). Возврат: (B, N, d_model).
    """
    def __init__(self, d_model: int, min_timescale: float = 1.0, max_timescale: float = 10_000.0):
        super().__init__()
        self.d_model = d_model
        self.min_timescale = float(min_timescale)
        self.max_timescale = float(max_timescale)

    def forward(self, ages_days: torch.Tensor) -> torch.Tensor:
        """
        ages_days: FloatTensor, shape (B, N). Значения >= 0 (0 — свежая новость).
        """
        b, n = ages_days.shape
        ages = ages_days.clamp(min=0.0).unsqueeze(-1)  # (B, N, 1)

        num_timescales = self.d_model // 2
        if num_timescales == 0:
            raise ValueError("d_model должен быть >= 2 для синусоидального кодирования.")

        # Геометрическая сетка частот
        log_inc = math.log(self.max_timescale / self.min_timescale) / max(num_timescales - 1, 1)
        timescales = self.min_timescale * torch.exp(
            torch.arange(num_timescales, device=ages.device, dtype=ages.dtype) * log_inc
        )  # (num_timescales,)

        angles = ages / timescales  # (B, N, num_timescales)
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, N, 2*num_timescales)

        # Если d_model нечётный — дополним нулём
        if pe.shape[-1] < self.d_model:
            pad = torch.zeros(b, n, 1, device=ages.device, dtype=ages.dtype)
            pe = torch.cat([pe, pad], dim=-1)
        return pe


class MergeBERT(nn.Module):
    """
    MergeBERT: принимает эмбеддинги последних N новостей и их давности,
    добавляет time encoding, прогоняет через 4 TransformerEncoderLayer и
    возвращает last hidden state размера (B, N, d_model).

    Параметры:
      d_model          — размерность входных эмбеддингов новостей
      nhead            — число голов внимания
      num_layers       — число encoder-блоков (по ТЗ — 4)
      dim_feedforward  — размер FFN в encoder-блоках
      dropout          — dropout в encoder-блоках и после суммы эмбеддингов
      min/max_timescale— диапазон частот для time encoding
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.05,
        min_timescale: float = 1.0,
        max_timescale: float = 10_000.0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers должно быть >= 1"

        self.d_model = d_model
        self.time_encoding = SinusoidalTimeEncoding(d_model, min_timescale, max_timescale)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # PyTorch encoder ожидает (S, B, E)
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.input_ln = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        news_embeddings: torch.Tensor,           # (B, N, d_model)
        ages_days: torch.Tensor,                 # (B, N)
        *,
        src_key_padding_mask = None,  # (B, N) True для PAD
        attn_mask = None              # (N, N)
    ) -> torch.Tensor:
        """
        Возвращает last hidden state: (B, N, d_model)
        """
        if news_embeddings.dim() != 3:
            raise ValueError("news_embeddings должен иметь форму (B, N, d_model)")
        if ages_days.shape[:2] != news_embeddings.shape[:2]:
            raise ValueError("ages_days и news_embeddings должны совпадать по (B, N)")
        if news_embeddings.size(-1) != self.d_model:
            raise ValueError(f"Последняя размерность news_embeddings должна быть {self.d_model}")

        # Добавим time encoding
        te = self.time_encoding(ages_days)                 # (B, N, d_model)
        x = self.input_ln(news_embeddings + te)            # (B, N, d_model)
        x = self.dropout(x)

        # TransformerEncoder: (S, B, E)
        x = x.transpose(0, 1)                              # (N, B, d_model)
        out = self.encoder(
            x,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask
        )                                                  # (N, B, d_model)

        return out.transpose(0, 1)                         # (B, N, d_model)

# %%
class Head(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.head(x)

# %%
def company_info_by_moex_ticker(ticker: str) -> dict:
    import requests, urllib.parse as u
    T=(ticker or "").upper().strip()
    if not T: raise ValueError("Укажите тикер, напр. 'SBER'.")
    j=lambda url,p=None: requests.get(url, params=p, timeout=12).json()

    m={}
    try:
        d=j(f"https://iss.moex.com/iss/securities/{u.quote(T)}.json")
        cs=d["description"]["columns"]; ni,vi=cs.index("name"),cs.index("value")
        m={r[ni]:r[vi] for r in d["description"]["data"]}
    except Exception:
        s=j("https://iss.moex.com/iss/securities.json", {"q":T})["securities"]; cs,rows=s["columns"],s["data"]
        gi=lambda k: cs.index(k) if k in cs else None
        row=next((r for r in rows if ((r[gi("SECID")] or "") if gi("SECID") is not None else "").upper()==T), rows[0] if rows else None)
        if not row: raise ValueError(f"Бумага {T!r} не найдена на MOEX.")
        m={k:(row[gi(k)] if gi(k) is not None else None) for k in ("SECID","SHORTNAME","NAME","EMITENT_TITLE","EMITENT_FULL_NAME","ISIN","TYPENAME","LISTLEVEL")}
    name=m.get("EMITENT_FULL_NAME") or m.get("EMITENT_TITLE") or m.get("NAME") or m.get("SHORTNAME") or T

    return name

# %%
# простой кеш на время запуска
_COMPANY_KW_CACHE = {}

def news_relates_to_company(text, ticker, threshold=2, model="openai/gpt-4o-mini", timeout=20):
    key = (ticker or "").strip().upper()
    if not key:
        return False, {"error": "пустой тикер"}

    # --- 1) Ключевые слова из кеша или через OpenRouter (RU) ---
    kws = _COMPANY_KW_CACHE.get(key)
    if isinstance(kws, int):
        if kws >= 4:
            return True
    if not isinstance(kws, list):
        company_name = company_info_by_moex_ticker(ticker)
        api_key = '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' # os.getenv("OPENROUTER_API_KEY")
        if api_key:
            sys_msg = (
                "Ты помощник, который делает КРАТКИЕ, но с высоким покрытием списки русскоязычных ключевых слов "
                "для сопоставления новостей с компаниями. Отвечай СТРОГО JSON без пояснений."
            )
            ctx = f"Тикер: {key}\n" \
                  + (f"Название компании: {company_name}\n" if company_name else "")
            user_msg = f"""
Верни ТОЛЬКО JSON вида {{"keywords": ["..."]}} (до 60 элементов) — список РУССКОЯЗЫЧНЫХ ключевых слов/фраз, по которым понятно, что новость относится к этой компании.

Включай:
1) Фирменные: официальные/разговорные/устаревшие названия, аббревиатуры, транслитерации (RU/EN), тикеры и их варианты.
2) Продукты/бренды/платформы; 1–3 часто упоминаемых руководителя (ФИО по-русски).
3) ОТРАСЛЕВЫЕ И МАКРО-МАРКЕРЫ для всей отрасли (чтобы ловить новости вида «банковский сектор вырос»). Добавь 10–20 характерных выражений и КОРНЕЙ (стем): часто употребимые части слов, покрывающие словоформы.
4) Гео-маркеры, если релевантно (страна/регион/город), частые варианты: "Россия","РФ" и т.п.

Требования:
- Для ключевых терминов давай КОРНИ (например "ипотек","платежн","карточн"), чтобы покрыть разные словоформы.
- Не добавляй общие слова типа "компания","рынок" без уникальной части (кроме отраслевых маркеров из п.3).
- Только JSON указанной формы.

Контекст (название тикера на МосБирже):
{ctx}""".strip()

            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model, "messages":[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
                          "temperature":0.1, "max_tokens":700},
                    timeout=timeout
                )
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                m = re.search(r"```json\s*(\{.*?\})\s*```", content, re.S) or re.search(r"(\{.*\})", content, re.S)
                obj = json.loads(m.group(1) if m else content)
                kws = obj.get("keywords", []) or [key]
            except Exception:
                if kws is None:
                    _COMPANY_KW_CACHE[key] = 1
                else:
                    _COMPANY_KW_CACHE[key] += 1
                return True
        else:
            kws = [key]

        # нормализация/дедуп + добавим тикер
        seen, cleaned = set(), []
        for k in (kws + [key]):
            if isinstance(k, str):
                s = k.strip()
                if s and s.lower() not in seen:
                    seen.add(s.lower()); cleaned.append(s)
        _COMPANY_KW_CACHE[key] = cleaned
        kws = cleaned

    # --- 2) Поиск с учётом словоформ ---
    # нормализуем ё→е (в тексте и в шаблонах)
    text = (text or "").replace("ё", "е")

    def _kw_regex(kw):
        kw = (kw or "").replace("ё", "е").strip()
        if not kw:
            return None
        # латиница/цифры — точное слово
        if re.search(r"[A-Za-z0-9]", kw):
            return re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)", re.I)
        # кириллица: добавляем суффикс [а-я]* к каждому слову, разделители — пробел/дефис
        parts = re.split(r"[\s\-]+", kw)
        parts = [(re.escape(p) + r"[а-я]*") if re.search(r"[А-Яа-я]", p) else re.escape(p) for p in parts if p]
        if not parts:
            return None
        sep = r'(?:[\s\-]+)'
        body = sep.join(parts)
        return re.compile(rf"(?<!\w){body}(?!\w)", re.I)

    matched, seen = [], set()
    for kw in kws:
        pat = _kw_regex(kw)
        if pat and pat.search(text):
            low = kw.lower()
            if low not in seen:
                seen.add(low); matched.append(kw)

    thr = max(1, int(threshold))
    return len(matched) >= thr#, {"ticker": key, "matched": matched, "match_count": len(matched), "threshold": thr, "keywords": kws}

# %%
def days_between(start, end, include_end: bool = False) -> int:
    """
    Кол-во календарных дней между двумя датами.
    Выходные считаются как обычные дни.

    Аргументы:
      start, end: datetime.date или datetime.datetime (порядок не важен).
      include_end: если True, конечная дата включается.

    Возвращает:
      Целое число дней.
    """
    # Приводим к date, если пришли datetime
    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()

    # Нормализуем порядок дат
    if end < start:
        start, end = end, start

    days = (end - start).days  # полуоткрытый интервал [start, end)
    if include_end:
        days += 1               # делаем интервал закрытым [start, end]

    return days

# %%
for ticker in candles['ticker']:
    news[ticker] = None
news = news.sort_values(by='publish_date', ascending=False).reset_index(drop=True)

# %%
n_articles_train = 16
n_articles_test = 16
n_train, n_test = 128, 16
dataset_train = []
dataset_test = []
for ticker in tqdm(candles['ticker'].unique(), desc='Preparing dataset...'):
    subset = candles[candles['ticker'] == ticker].copy()
    subset = subset.sort_values(by='begin', ascending=False).reset_index(drop=True)
    for idx in (subset.index[:n_test]):
        now = subset.loc[idx, 'begin']
        news_before = news[(now - timedelta(days=100) < news['publish_date']) & \
                           (news['publish_date'] < now + timedelta(days=1))].reset_index(drop=True)
        articles = []
        times = []
        for i in news_before.index:
            if pd.isna(news_before.loc[i, ticker]):
                news_before.loc[i, ticker] = news_relates_to_company(news_before.loc[i, 'title'], ticker)
                # if not news_before.loc[i, ticker]:
                #     news_before.loc[i, ticker] = news_relates_to_company(news_before.loc[i, 'publication'], ticker)
            if news_before.loc[i, ticker]:
                d = news_before.loc[i, 'publish_date']
                articles.append(news_before.loc[i, 'publication'],)
                times.append(float(days_between(datetime.date(d), datetime.date(now))))
            if len(articles) >= n_articles_train:
                break
        if len(articles) < n_articles_test:
            articles += ['' for i in range(n_articles_test - len(articles))]
            times += [100.0 for i in range(n_articles_test - len(times))]
        dataset_test.append((articles, times, ticker, subset.loc[idx, 'train_target']))
    for idx in (subset.index[n_test + 32:n_test + 32 + n_train]):
        now = subset.loc[idx, 'begin']
        news_before = news[(now - timedelta(days=100) < news['publish_date']) & \
                           (news['publish_date'] < now + timedelta(days=1))].reset_index(drop=True)
        articles = []
        times = []
        for i in news_before.index:
            if pd.isna(news_before.loc[i, ticker]):
                news_before.loc[i, ticker] = news_relates_to_company(news_before.loc[i, 'title'], ticker)
                # if not news_before.loc[i, ticker]:
                #     news_before.loc[i, ticker] = news_relates_to_company(news_before.loc[i, 'publication'], ticker)
            if news_before.loc[i, ticker]:
                d = news_before.loc[i, 'publish_date']
                articles.append(news_before.loc[i, 'publication'],)
                times.append(float(days_between(datetime.date(d), datetime.date(now))))
            if len(articles) >= n_articles_test:
                break
        if len(articles) < n_articles_train:
            articles += ['' for i in range(n_articles_train - len(articles))]
            times += [100.0 for i in range(n_articles_train - len(times))]
        dataset_train.append((articles, times, ticker, subset.loc[idx, 'train_target']))
print()

# %%
import json
with open('cache.json', 'w') as f:
    json.dump((dataset_train, dataset_test), f)

# %%
import json
with open('cache.json', 'r') as f:
    dataset_train, dataset_test = json.load(f)

# %%
target_train = pd.Series([i[3] for i in dataset_train])
target_test = pd.Series([i[3] for i in dataset_test])

print('Train:')
print(f'Const Acc (val):    {np.mean((target_train >= 0) == (target_train >= 0).mode()[0]):.6f}')
print(f'Const MAE (val):    {np.mean(np.abs(target_train - target_train.median())):.6f}')
print(f'Const RMSE (val):   {np.sqrt(np.mean(np.square(target_train - target_train.mean()))):.6f}')
print(f'Const MSE (val):    {np.mean(np.square(target_train - target_train.mean())):.6f}')
print()

print('Val:')
print(f'Const Acc (val):    {np.mean((target_test >= 0) == (target_train >= 0).mode()[0]):.6f}')
print(f'Const MAE (val):    {np.mean(np.abs(target_test - target_train.median())):.6f}')
print(f'Const RMSE (val):   {np.sqrt(np.mean(np.square(target_test - target_train.mean()))):.6f}')
print(f'Const MSE (val):    {np.mean(np.square(target_test - target_train.mean())):.6f}')
print()

# %%
tickers = candles['ticker'].unique().tolist()
le = LabelEncoder()
le.fit(tickers)
ticker_embedding = nn.Embedding(len(le.classes_), hidden_dim2)

# %%
class Model(nn.Module):
    def __init__(self, finbert, connector, ticker_embedding, mergebert, head):
        super().__init__()
        self.finbert = finbert
        self.connector = connector
        self.ticker_embedding = ticker_embedding
        self.mergebert = mergebert
        self.head = head

    def embed(self, articles, times, ticker):
        # articles: (B, N, L)
        s = articles['input_ids'].shape
        embeds = self.finbert(input_ids=articles['input_ids'].view(-1, s[-1]),
                            attention_mask=articles['attention_mask'].view(-1, s[-1])).last_hidden_state.mean(dim=1)
        embeds = self.connector(embeds)
        embeds = embeds.view(s[0], s[1], embeds.shape[-1])
        ticker_embed = self.ticker_embedding(ticker).unsqueeze(1)
        embeds = torch.cat([embeds, ticker_embed], dim=1)
        times = torch.cat([times, torch.zeros(s[0], 1).to(device, dtype)], dim=-1)
        return self.mergebert(embeds, times)[:, -1, :]

    def forward(self, articles, times, ticker):
        embeds = self.embed(articles, times, ticker)
        return self.head(embeds)

# %%
model = Model(
    finbert.bert,
    Connector(hidden_dim1, hidden_dim2),
    ticker_embedding,
    MergeBERT(hidden_dim2),
    Head(hidden_dim2),
).to(device, dtype)

for p in model.finbert.embeddings.parameters():
    p.requires_grad = False
for p in model.finbert.encoder.layer[0].parameters():
    p.requires_grad = False

# %%
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, a, tokenizer, ticker_encoder):
        self.a = a
        self.tokenizer = tokenizer
        self.ticker_encoder = ticker_encoder

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        # (articles, times, ticker, train_target)
        articles_embeds = tokenizer(
            self.a[idx][0],
            max_length=512, padding='max_length', truncation=True, return_tensors='pt'
        )
        ticker_encoding = self.ticker_encoder.transform([self.a[idx][2]])[0]
        return articles_embeds, torch.tensor(self.a[idx][1]), ticker_encoding, torch.tensor(self.a[idx][3])

# %%
train_set, val_set = dataset_train, dataset_test # train_test_split(dataset, test_size=0.1, shuffle=True, random_state=SEED)

train_set = CustomDataset(train_set, tokenizer, le)
val_set = CustomDataset(val_set, tokenizer, le)

# %%
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False)

# %%
loss_fn = nn.SmoothL1Loss(beta=candles['train_target'].std() / 4)

n_epochs = 6
grad_accum_steps = 1
warmup_ratio = 0.1
lrs = (1e-7, 3e-4, 1e-7)
weight_decay = 3e-4

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=lrs[1], weight_decay=weight_decay)

total_steps = n_epochs * len(train_loader)
warmup_iters = int(round(total_steps * warmup_ratio, 0))
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=lrs[0] / lrs[1], total_iters=warmup_iters)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps - warmup_iters, lrs[2])
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])

# %%
losses = []
for epoch in range(n_epochs):
    print(f'Epoch {epoch + 1}/{n_epochs}')

    pb = tqdm(enumerate(train_loader), total=len(train_loader))
    optimizer.zero_grad()
    loss = torch.tensor(0.0).to(device, dtype)
    p, t = torch.tensor([]), torch.tensor([])
    for i, (articles, times, ticker, y) in pb:
        articles = articles.to(device)
        times = times.to(device, dtype)
        ticker = ticker.to(device)
        y = y.to(device, dtype)

        model.train()
        preds = model(articles, times, ticker)

        p = torch.cat([p, preds.detach().view(-1).cpu()])
        t = torch.cat([t, y.cpu()])

        loss = loss + loss_fn(preds, y.view(-1, 1))
        if (i + 1) % grad_accum_steps == 0 or i + 1 == len(train_loader):
            loss = loss / ((i + 1) - (i // grad_accum_steps) * grad_accum_steps)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            pb.set_description(f'Training Loss: {loss.item():.6f}')
            loss = torch.tensor(0.0).to(device, dtype)
        scheduler.step()

        if i + 1 == len(train_loader):
            train_loss = loss_fn(p, t)
            train_mae = nn.L1Loss()(p, t)
            train_mse = nn.MSELoss()(p, t)
            train_acc = torch.mean(((p >= 0.0) == (t >= 0.0)).float())
            print(f'Train (step {i})\t | Loss: {train_loss:.6f} | MAE: {train_mae.item():.6f} | RMSE: {torch.sqrt(train_mse).item():.6f} | MSE: {train_mse.item():.6f} | Acc: {train_acc:.6f}')

        if i + 1 == len(train_loader): # (i + 1) % 256 == 0 or i == 0 or
            model.eval()
            with torch.no_grad():
                pb2 = tqdm(val_loader, leave=False)
                p, t = torch.tensor([]), torch.tensor([])
                for articles, times, ticker, y in pb2:
                    articles = articles.to(device)
                    times = times.to(device, dtype)
                    ticker = ticker.to(device)

                    p = torch.cat([p, model(articles, times, ticker).view(-1).cpu()])
                    t = torch.cat([t, y.cpu()])

                val_loss = loss_fn(p, t)
                val_mae = nn.L1Loss()(p, t)
                val_mse = nn.MSELoss()(p, t)
                val_acc = torch.mean(((p >= 0.0) == (t >= 0.0)).float())
                print(f'Val (step {i})  \t | Loss: {val_loss:.6f} | MAE: {val_mae.item():.6f} | RMSE: {torch.sqrt(val_mse).item():.6f} | MSE: {val_mse.item():.6f} | Acc: {val_acc:.6f}')
    print()

# %%
def get_embedding(news: pd.DataFrame, date: datetime, ticker: str, n_articles=16) -> np.ndarray:
    context = news[(date - timedelta(days=100) < news['publish_date']) & \
                   (news['publish_date'] < date + timedelta(days=1))]
    context = context.sort_values(by='publish_date', ascending=False).reset_index(drop=True)
    articles, times = [], []
    for i in context.index:
        if pd.isna(context.loc[i, ticker]):
            context.loc[i, ticker] = news_relates_to_company(context.loc[i, 'title'], ticker)
        if context.loc[i, ticker]:
            articles.append(context.loc[i, 'publication'],)
            times.append(float(days_between(datetime.date(context.loc[i, 'publish_date']),
                                            datetime.date(date))))
        if len(articles) >= n_articles:
            break
    if len(articles) < n_articles:
        articles += ['' for i in range(n_articles - len(articles))]
        times += [50.0 for i in range(n_articles - len(times))]

    with torch.no_grad():
        articles = tokenizer(
            articles,
            max_length=512, padding='max_length', truncation=True, return_tensors='pt'
        )
        times = torch.tensor(times)
        ticker = torch.tensor(le.transform([ticker])[0])

        embeds = model.embed(
            {
                'input_ids': articles['input_ids'].to(device).unsqueeze(0),
                'attention_mask': articles['attention_mask'].to(device).unsqueeze(0)
            },
            times.to(device).unsqueeze(0),
            ticker.to(device).unsqueeze(0)
        ).cpu()[0]
    return embeds.float().numpy()

# %%
candles = pd.concat([pd.read_csv('data/candles.csv'), pd.read_csv('data/candles_2.csv')])
candles['begin'] = pd.to_datetime(candles['begin'])
candles = candles.drop_duplicates().sort_values(by=['ticker', 'begin']).reset_index(drop=True)

# %%
n = 128
dataset = []
for ticker in tqdm(candles['ticker'].unique(), desc='Generating embeddings for train...'):
    subset = candles[(candles['ticker'] == ticker) & (candles['begin'] <= last_date - timedelta(days=20))].copy()
    subset = subset.sort_values(by='begin', ascending=False)
    for idx in (subset.index[:n]):
        dataset.append((idx, get_embedding(news, candles.loc[idx, 'begin'], ticker)))

# %%
for ticker in tqdm(candles['ticker'].unique(), desc='Generating embeddings for test...'):
    subset = candles[(candles['ticker'] == ticker) & (candles['begin'] == last_date)].copy()
    for idx in (subset.index):
        dataset.append((idx, get_embedding(news, candles.loc[idx, 'begin'], ticker)))

# %%
def compress_embeddings_with_pca(pairs, n_components, batch_size=10_000, dtype=np.float32):
    if not pairs:
        return []

    if len(pairs) < n_components:
        raise ValueError(
            f"Для PCA с {n_components} компонентами нужно минимум {n_components} образцов, "
            f"а получено {len(pairs)}."
        )

    ipca = IncrementalPCA(n_components=n_components)

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        X = np.stack([emb.astype(dtype, copy=False) for _, emb in batch], axis=0)
        ipca.partial_fit(X)

    reduced_pairs = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        X = np.stack([emb.astype(dtype, copy=False) for _, emb in batch], axis=0)
        Xr = ipca.transform(X).astype(dtype, copy=False)
        reduced_pairs.extend((idx, Xr[i]) for i, (idx, _) in enumerate(batch))

    return reduced_pairs

# %%
reduced_size = 32
dataset_reduced = compress_embeddings_with_pca(dataset, n_components=reduced_size)

# %%
emb_features = [f'emb_{i}' for i in range(reduced_size)]
for f in emb_features:
    candles[f] = np.nan

for idx, emb in tqdm(dataset_reduced, desc='Building dataset...'):
    candles.loc[idx, emb_features] = emb

# %%
candles.to_parquet('candles_with_embs.parquet', index=False)

# %%



