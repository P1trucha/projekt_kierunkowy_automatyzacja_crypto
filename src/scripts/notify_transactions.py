#!/usr/bin/env python3
import os
import json
import csv
import io
import smtplib
from email.message import EmailMessage
from collections import defaultdict

CSV_PATH = os.getenv("TX_CSV_PATH", "logs/transactions.csv")
STATE_PATH = os.getenv("TX_STATE_PATH", "logs/.tx_notify_state.json")

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
MAIL_TO = os.getenv("MAIL_TO", "")
MAIL_FROM = os.getenv("MAIL_FROM", SMTP_USER)

# Tylko te eventy wysyłamy mailem
TX_EVENTS = [e.strip() for e in os.getenv("TX_EVENTS", "TRADE_ENTRY,TRADE_EXIT").split(",") if e.strip()]

MAX_ROWS_PER_EMAIL = int(os.getenv("TX_MAX_ROWS_PER_EMAIL", "50"))
SUBJECT_PREFIX = os.getenv("TX_SUBJECT_PREFIX", "[binancebot]")

def load_state():
    if not os.path.exists(STATE_PATH):
        return {"last_offset": 0}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            st = json.load(f) or {}
        return {"last_offset": int(st.get("last_offset", 0))}
    except Exception:
        return {"last_offset": 0}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)

def read_new_chunk(last_offset: int):
    """
    Zwraca (text_chunk, new_offset, file_size).
    Jeśli plik się skrócił (rotacja/truncate), resetujemy offset.
    """
    if not os.path.exists(CSV_PATH):
        return "", 0, 0

    size = os.path.getsize(CSV_PATH)
    if last_offset > size:
        last_offset = 0

    with open(CSV_PATH, "rb") as f:
        f.seek(last_offset)
        data = f.read()
        new_offset = f.tell()

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        text = data.decode(errors="replace")

    return text, new_offset, size

def parse_csv_rows(text_chunk: str):
    """
    Parsuje dopisany fragment CSV. Obsługuje header jeśli chunk zaczyna się od niego.
    Zwraca listę dictów (kolumny wg headera).
    """
    if not text_chunk.strip():
        return []

    # Ujednolicenie końcówek linii
    text_chunk = text_chunk.replace("\r\n", "\n").replace("\r", "\n")

    # csv.DictReader potrzebuje IO z pełnymi liniami
    buf = io.StringIO(text_chunk)

    # Spróbuj wykryć czy w tym kawałku jest header
    # Jeśli bot dopisuje bez headera (typowo), to DictReader bez fieldnames nie ruszy.
    first_line = buf.readline()
    if not first_line:
        return []

    # Cofnij wskaźnik
    buf.seek(0)

    # Jeśli to header (zawiera "event" i "symbol"), czytamy normalnie
    if "event" in first_line and "symbol" in first_line:
        reader = csv.DictReader(buf)
        return [row for row in reader if row]
    else:
        # Brak headera w chunku -> musimy wziąć header z pliku
        if not os.path.exists(CSV_PATH):
            return []

        with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
            header = f.readline().strip()

        if not header:
            return []

        fieldnames = [h.strip() for h in header.split(",")]
        reader = csv.DictReader(buf, fieldnames=fieldnames)

        rows = []
        for row in reader:
            # jeśli trafiłby się header w środku, omiń
            if row.get("event") == "event":
                continue
            rows.append(row)
        return rows

def only_trade_events(rows):
    out = []
    for r in rows:
        ev = (r.get("event") or "").strip()
        if ev in TX_EVENTS:
            out.append(r)
    return out

def format_email(rows):
    """
    Buduje czytelny mail: grupuje po trade_id (jeśli jest), pokazuje ENTRY/EXIT.
    """
    # Grupowanie
    grouped = defaultdict(list)
    for r in rows:
        trade_id = (r.get("trade_id") or r.get("run_id") or "no_id").strip()
        grouped[trade_id].append(r)

    lines = []
    lines.append(f"CSV: {CSV_PATH}")
    lines.append(f"Nowe zdarzenia (filtrowane): {len(rows)}")
    lines.append(f"Filtr eventów: {', '.join(TX_EVENTS)}")
    lines.append("")
    lines.append("Szczegóły:")
    lines.append("------------------------------------------------------------")

    # Stabilne sortowanie: najpierw ENTRY potem EXIT, w obrębie po ts_epoch
    def sort_key(r):
        ev = (r.get("event") or "")
        pri = 0 if ev.endswith("ENTRY") else 1
        try:
            ts = float(r.get("ts_epoch") or 0)
        except Exception:
            ts = 0
        return (pri, ts)

    for tid, items in grouped.items():
        items = sorted(items, key=sort_key)
        lines.append(f"\ntrade_id={tid}")

        for r in items:
            ev = (r.get("event") or "").strip()
            symbol = (r.get("symbol") or "").strip()
            side = (r.get("side") or "").strip()
            qty = (r.get("qty") or "").strip()
            entry = (r.get("entry_price") or "").strip()
            exitp = (r.get("exit_price") or "").strip()
            pnl = (r.get("pnl_pct") or "").strip()
            hold = (r.get("hold_sec") or "").strip()
            reason = (r.get("reason") or "").strip()
            ts = (r.get("ts_utc") or r.get("ts_local") or "").strip()

            if ev == "TRADE_ENTRY":
                lines.append(f"  - {ts} ENTRY  {symbol} {side} qty={qty} entry={entry}")
            elif ev == "TRADE_EXIT":
                extra = []
                if pnl != "":
                    extra.append(f"pnl_pct={pnl}")
                if hold != "":
                    extra.append(f"hold_sec={hold}")
                if reason != "":
                    extra.append(f"reason={reason}")
                extra_s = (" | " + " ".join(extra)) if extra else ""
                lines.append(f"  - {ts} EXIT   {symbol} {side} qty={qty} exit={exitp}{extra_s}")
            else:
                lines.append(f"  - {ts} {ev} {symbol}")

    return "\n".join(lines)

def send_email(subject, body):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and MAIL_TO):
        raise RuntimeError("Brak konfiguracji SMTP (SMTP_HOST/SMTP_USER/SMTP_PASS/MAIL_TO).")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = MAIL_FROM
    msg["To"] = MAIL_TO
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=25) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

def main():
    state = load_state()
    last_offset = int(state.get("last_offset", 0))

    chunk, new_offset, size = read_new_chunk(last_offset)

    # Nic nie dopisano -> nic nie rób
    if not chunk:
        return

    rows = parse_csv_rows(chunk)
    trade_rows = only_trade_events(rows)

    # Zawsze aktualizujemy offset, żeby nie wysyłać w kółko tego samego
    state["last_offset"] = new_offset
    save_state(state)

    # Jeśli nic z ENTRY/EXIT nie ma -> brak maila
    if not trade_rows:
        return

    # Limit rekordów na maila
    clipped = False
    if len(trade_rows) > MAX_ROWS_PER_EMAIL:
        trade_rows = trade_rows[-MAX_ROWS_PER_EMAIL:]
        clipped = True

    body = format_email(trade_rows)
    if clipped:
        body += f"\n\n(Ucięto do ostatnich {MAX_ROWS_PER_EMAIL} zdarzeń)"

    subject = f"{SUBJECT_PREFIX} transactions: {len(trade_rows)} new events"
    send_email(subject, body)

if __name__ == "__main__":
    main()