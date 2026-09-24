# rag_qdrant

`rag_qdrant` to narzędzie do indeksowania dokumentacji Markdown w lokalnej bazie [Qdrant](https://qdrant.tech/) i wykonywania błyskawicznego wyszukiwania semantycznego.

Od wersji 2.0 architektura opiera się na **jednym trwałym procesie w tle**, który trzyma model embeddingów FastEmbed (`BAAI/bge-base-en-v1.5`) i połączenie z bazą Qdrant w pamięci RAM/VRAM. Zarówno cienki klient CLI `rag_qdrant`, jak i serwer MCP (dla agentów AI) korzystają z tego samego silnika, eliminując koszt ponownego importowania bibliotek i ładowania wag przy każdym zapytaniu.

---

## Główne cechy

- **Błyskawiczne wyszukiwanie**: Czas odpowiedzi CLI spadł z ~2.1 s do ~0.15–0.20 s.
- **Pełna kompatybilność wsteczna**: Wszystkie dotychczasowe polecenia, flagi (`--status`, `--list-sources`, `search`, `--json`, `--index-json`) oraz skrypty `rag_qdrant.bat` i `rag_qdrant.ps1` działają dokładnie tak samo.
- **Autostart na żądanie**: Jeśli serwis w tle nie działa, CLI uruchamia go automatycznie, czeka na gotowość i natychmiast wykonuje żądanie.
- **Podwójny interfejs**: Lokalne REST API oraz serwer MCP (Streamable HTTP / SSE) na wspólnym porcie `127.0.0.1:6335`.
- **Wysokie bezpieczeństwo**: Serwis i Qdrant nasłuchują wyłącznie na pętli zwrotnej (`127.0.0.1`), nagłówki `Host` i `Origin` są walidowane przeciwko atakom DNS rebinding, a dostęp do danych wymaga tokenu.
- **Profile uprawnień**: Profile klientów (`admin`, `amiga`, `devnotes`) ograniczają zakres widocznych i indeksowanych źródeł oraz katalogów.

---

## Instalacja

W katalogu `rag-qdrant`:

```powershell
pip install -r requirements.txt
```

Opcjonalnie dla akceleracji GPU (NVIDIA RTX / CUDA):
```powershell
pip install onnxruntime-gpu nvidia-cublas-cu12
```

---

## Konfiguracja (`.env`)

Utwórz lub zaktualizuj plik `.env` w katalogu głównym lub w `rag-qdrant/`:

```env
QDRANT_API_KEY=twoj-klucz-api-qdrant
QDRANT_URL=http://127.0.0.1:6333

RAG_SERVICE_HOST=127.0.0.1
RAG_SERVICE_PORT=6335

RAG_ADMIN_TOKEN=losowy-token-administratora
RAG_AMIGA_TOKEN=losowy-token-profilu-amiga
RAG_DEVNOTES_TOKEN=losowy-token-profilu-devnotes

RAG_CANONICAL_INDEX_JSON=d:\AI\qdrant\amiga_rag_cache.json
```

---

## Polecenia CLI

Uruchamiaj CLI przez `bin\rag_qdrant.ps1` lub `bin\rag_qdrant.bat`, albo jako moduł Python: `python -m rag_qdrant.cli`.

### 1. Wyszukiwanie semantyczne

```powershell
rag_qdrant search "DMA arbitration" --source amiga --limit 5 --index-json D:\AI\qdrant\amiga_rag_cache.json --json
```

| Parametr | Opis |
| --- | --- |
| `ZAPYTANIE` | Tekst zapytania w języku naturalnym |
| `-s TAGI`, `--source TAGI` | Opcjonalny tag źródła lub tagi po przecinku (np. `amiga,devnotes`) |
| `--limit N` | Maksymalna liczba wyników (domyślnie 5, min 1, maks 50) |
| `--index-json PLIK` | Wymagany plik JSON stanu indeksu |
| `--json` | Wymagany. Zwraca tablicę obiektów z polami `score`, `source`, `file_path`, `relative_path`, `header`, `content`, `images`. |

### 2. Stan kolekcji i źródeł

```powershell
# Stan bazy wektorowej i kolekcji
rag_qdrant --status --index-json D:\AI\qdrant\amiga_rag_cache.json [--json]

# Lista zaindeksowanych źródeł
rag_qdrant --list-sources --index-json D:\AI\qdrant\amiga_rag_cache.json [--json]
```

### 3. Indeksowanie katalogu

```powershell
rag_qdrant D:\Docs\Amiga --source amiga --index-json D:\AI\qdrant\amiga_rag_cache.json
```

- Skanuje pliki Markdown pod wskazanym katalogiem i oblicza ich hashe SHA-256.
- Nowe i zmodyfikowane pliki są dzielone na fragmenty, wektoryzowane i zapisywane w Qdrant.
- Niezmienione pliki są pomijane (0 kosztu wektoryzacji).
- Pliki usunięte z dysku są automatycznie usuwane z Qdrant i pliku stanu.
- Wyświetla na żywo pasek postępu i plan indeksowania przez strumieniowane SSE z serwisu.

### 4. Zarządzanie serwisem w tle

```powershell
rag_qdrant service status   # Sprawdza czy serwis działa i podaje statystyki
rag_qdrant service start    # Ręcznie uruchamia proces serwisu w tle
rag_qdrant service stop     # Zatrzymuje działający proces serwisu
```

---

## Lokalne REST API (`http://127.0.0.1:6335`)

Wszystkie endpointy poza `/v1/health` i `/v1/ready` wymagają nagłówka `Authorization: Bearer <TOKEN>` lub `X-API-Key: <TOKEN>`.

| Metoda | Ścieżka | Opis |
| --- | --- | --- |
| `GET` | `/v1/health` | Sprawdzenie liveness serwisu (`{"status": "ok"}`) |
| `GET` | `/v1/ready` | Sprawdzenie gotowości modelu i kolekcji |
| `GET` | `/v1/status` | Statystyki Qdrant i kolekcji |
| `GET` | `/v1/sources` | Lista źródeł, liczba plików i fragmentów |
| `POST` | `/v1/search` | JSON: `{"query": "...", "source": "...", "limit": 5}` |
| `POST` | `/v1/index` | JSON: `{"path": "...", "source": "...", "stream": false/true}` |
| `POST` | `/v1/service/stop` | Czyste zatrzymanie procesu (wymaga profilu admin) |

---

## Serwer MCP (`http://127.0.0.1:6335/mcp`)

Serwis udostępnia wbudowany serwer MCP zgodny ze standardem Streamable HTTP / SSE.

### Dostępne narzędzia:
- `search(query, sources=None, limit=5)`: Wyszukiwanie semantyczne z uwzględnieniem uprawnień profilu tokenu.
- `status()`: Odczyt stanu kolekcji bez modyfikacji.
- `list_sources()`: Statystyki źródeł z cache.
- `index(path, source)`: Przyrostowe indeksowanie (dozwolone tylko dla tokenów z uprawnieniem zapisu).

### Konfiguracja klienta MCP (np. Antigravity IDE, Claude Desktop, Cursor):

W pliku `mcp_config.json`:
```json
{
  "mcpServers": {
    "qdrant-rag": {
      "url": "http://127.0.0.1:6335/mcp?token=TWOJ_TOKEN_PROFILU",
      "transport": "sse"
    }
  }
}
```

---

## Architektura i pliki

```
rag-qdrant/
├── bin/
│   ├── rag_qdrant.bat       # Launcher CMD dla Windows
│   └── rag_qdrant.ps1       # Launcher PowerShell dla Windows
├── rag_qdrant/
│   ├── arguments.py         # Parsowanie argumentów CLI
│   ├── cache.py             # Normalizacja i schemat pliku JSON stanu
│   ├── chunker.py           # Dzielenie plików Markdown na logiczne sekcje
│   ├── client.py            # Cienki klient HTTP z obsługą autostartu i SSE
│   ├── cli.py               # Główny punkt wejścia CLI
│   ├── config.py            # Ustawienia, środowisko i parametry sprzętowe
│   ├── core.py              # Centralny silnik RagEngine (Qdrant + FastEmbed + Lock)
│   ├── discovery.py         # Wykrywanie plików .md z ignorowaniem katalogów prywatnych
│   ├── indexer.py           # Kompatybilna klasa KnowledgeIndexer
│   ├── security.py          # Walidacja Host/Origin, uwierzytelnianie i profile
│   └── service.py           # Serwis Starlette/Uvicorn + FastMCP
└── tests/
    ├── test_indexing_contract.py  # Testy kontraktu CLI i cache
    └── test_service.py            # Testy REST API, bezpieczeństwa i MCP
```
