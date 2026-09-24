# Qdrant Vector Database & Persistent RAG Service

Ten katalog zawiera konfigurację Docker Compose dla bazy wektorowej [Qdrant](https://qdrant.tech/), pamięć masową wektorów oraz centralny serwis `rag_qdrant` obsługujący semantyczne wyszukiwanie, indeksowanie dokumentacji, REST API i serwer MCP.

---

## 1. Architektura systemu

```text
rag_qdrant CLI ── HTTP/JSON ──┐
                              ├── proces serwisu (127.0.0.1:6335) ── Qdrant Docker (127.0.0.1:6333)
klient MCP ── Streamable HTTP ┘       │
                                      └── model FastEmbed i stan indeksu w RAM/VRAM
```

- **Jeden proces w tle**: Utrzymuje model embeddingów (`BAAI/bge-base-en-v1.5` z akceleracją CUDA/CPU) oraz klienta Qdrant w pamięci.
- **Dwa wejścia**:
  1. Lokalne REST API HTTP/JSON dla cienkiego klienta CLI `rag_qdrant`.
  2. Serwer MCP (Streamable HTTP / SSE) na ścieżce `/mcp` dla agentów AI.
- **Cienki klient CLI**: Nie ładuje modelu ani ciężkich bibliotek przy każdym zapytaniu. Czas odpowiedzi wyszukiwania spadł z ~2.1 s do ~0.15–0.20 s. W razie potrzeby klient uruchamia serwis w tle automatycznie na żądanie.

---

## 2. Bezpieczeństwo i porty

1. **Izolacja sieciowa (Loopback only)**:
   - Qdrant nasłuchuje wyłącznie na `127.0.0.1:6333` (REST) i `127.0.0.1:6334` (gRPC).
   - Serwis `rag_qdrant` nasłuchuje wyłącznie na `127.0.0.1:6335`.
2. **Klucz API Qdrant**:
   - Kontener Qdrant wymaga klucza `QDRANT__SERVICE__API_KEY` pobieranego z pliku `.env`.
   - Żądania bez klucza do `http://127.0.0.1:6333/collections` są odrzucane z kodem HTTP 401 Unauthorized.
3. **Uwierzytelnianie serwisu RAG**:
   - Wszystkie endpointy danych (REST oraz `/mcp`) wymagają tokenu autoryzacyjnego (`Authorization: Bearer <TOKEN>` lub `?token=<TOKEN>`).
   - Profile klientów (`admin`, `amiga`, `devnotes`) definiują dozwolone źródła wyszukiwania, źródła indeksowania i dozwolone katalogi.
4. **Ochrona przed DNS Rebinding**:
   - Serwis waliduje nagłówki `Host` i `Origin`. Zapytania z obcymi wartościami (np. próby ataku ze skryptów w przeglądarce) są natychmiast odrzucane z kodem HTTP 403 Forbidden.

---

## 3. Konfiguracja (`.env`)

Skopiuj `.env.example` do `.env` (plik `.env` jest ignorowany w `.gitignore`):

```bash
# Qdrant Docker
QDRANT_API_KEY=twoj-losowy-klucz-api-dla-qdrant
QDRANT_URL=http://127.0.0.1:6333

# Serwis rag-qdrant
RAG_SERVICE_HOST=127.0.0.1
RAG_SERVICE_PORT=6335

# Tokeny uwierzytelniające
RAG_ADMIN_TOKEN=losowy-token-dla-cli-i-admina
RAG_AMIGA_TOKEN=losowy-token-profilu-amiga
RAG_DEVNOTES_TOKEN=losowy-token-profilu-devnotes

# Wspólny stan lokalny kolekcji
RAG_CANONICAL_INDEX_JSON=d:\AI\qdrant\amiga_rag_cache.json
```

---

## 4. Zarządzanie bazą Qdrant

Uruchamiaj z katalogu głównego projektu:

```powershell
# Uruchomienie kontenera w tle
docker compose up -d

# Logi kontenera
docker compose logs -f

# Sprawdzenie gotowości
curl http://127.0.0.1:6333/readyz

# Zatrzymanie kontenera
docker compose down
```

---

## 5. Zarządzanie serwisem RAG

Serwis uruchamia się **automatycznie na żądanie** przy pierwszym wywołaniu `rag_qdrant`. Można nim też zarządzać bezpośrednio:

```powershell
# Stan serwisu
rag_qdrant service status

# Ręczny start w tle
rag_qdrant service start

# Zatrzymanie serwisu
rag_qdrant service stop
```

---

## 6. Serwer MCP dla agentów AI

Serwis wystawia serwer MCP pod adresem:
`http://127.0.0.1:6335/mcp?token=<RAG_ADMIN_TOKEN_LUB_PROFIL>`

### Narzędzia MCP:
- `search(query: str, sources: Optional[str] = None, limit: int = 5)`: Wyszukiwanie semantyczne w dokumentacji z uwzględnieniem ograniczeń profilu.
- `status()`: Stan bazy wektorowej, kolekcji `projects_docs` i liczby wektorów.
- `list_sources()`: Lista zaindeksowanych źródeł, liczba plików i fragmentów.
- `index(path: str, source: str)`: Przyrostowe indeksowanie katalogu (wymaga uprawnienia do zapisu).

### Konfiguracja w `mcp_config.json`:

Dla klienta obsługującego transport SSE/HTTP:
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

Dla dotychczasowych adapterów MCP uruchamiających CLI:
Wszystkie dotychczasowe polecenia `rag_qdrant.bat` i `rag_qdrant.ps1` działają bez zmian, ale zyskują 10–20-krotnie szybsze wykonywanie zapytań dzięki odpytywaniu stałego serwisu.

---

## 7. Struktura katalogu

```
.
├── README.md                 # Niniejsza dokumentacja
├── docker-compose.yml        # Konfiguracja kontenera Qdrant z izolacją 127.0.0.1
├── .env.example              # Szablon konfiguracji i sekretów
├── amiga_rag_cache.json      # Wspólny plik stanu indeksu
├── qdrant_storage/           # Wolumen z danymi Qdrant
└── rag-qdrant/               # Pakiet kodu serwisu i CLI
    ├── bin/                  # Launchery rag_qdrant.bat i rag_qdrant.ps1
    ├── rag_qdrant/           # Implementacja core, service, client, cli, security
    └── tests/                # Testy jednostkowe i regresyjne kontraktu
```
