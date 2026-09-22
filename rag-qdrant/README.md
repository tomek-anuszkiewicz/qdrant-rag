# rag_qdrant

`rag_qdrant` jest lokalnym narzędziem CLI do indeksowania dokumentacji Markdown
w Qdrant i wyszukiwania semantycznego. Kolekcja Qdrant może zawierać wiele
źródeł oznaczonych tagiem `--source`. Narzędzie używa lokalnych embeddingów
FastEmbed; Qdrant powinien być dostępny pod `http://localhost:6333`.

Podczas indeksowania skanowane są wszystkie pliki `.md` w `PATH`, łącznie z
plikami w katalogu głównym i we wszystkich podkatalogach. Pomijane są tylko
wbudowane katalogi techniczne i prywatne: m.in. `.git`, `.obsidian`, `.venv`,
`node_modules`, `__pycache__` oraz katalogi, których nazwa zawiera `private`.

## Instalacja i konfiguracja

W katalogu `rag-qdrant` zainstaluj zależności:

```powershell
pip install -r requirements.txt
```

Każde polecenie poza `--help` wymaga `--index-json ŚCIEŻKA`. Jest to jawnie
wskazany plik JSON lokalnego stanu indeksu: zawiera hashe SHA-256 plików,
liczbę fragmentów, źródła i metadane obrazów. Nie zawiera wektorów — te są w
Qdrant. Gdy plik nie istnieje, zostanie utworzony przy pierwszym zapisie.
Do kolejnych przebiegów tej samej kolekcji przekazuj zawsze ten sam plik.

Uruchamiaj CLI przez `bin\rag_qdrant.ps1` lub `bin\rag_qdrant.bat`, albo po
dodaniu katalogu do `PATH` jako `rag_qdrant`.

## Polecenia i parametry

### Indeksowanie

```text
rag_qdrant PATH --source NAZWA --index-json PLIK
```

| Parametr | Znaczenie |
| --- | --- |
| `PATH` | Wymagana pozycyjna ścieżka do katalogu dokumentów. Wszystkie kwalifikujące się pliki Markdown pod tym katalogiem są skanowane. |
| `-s NAZWA`, `--source NAZWA` | Wymagany tag źródła, np. `project-a`. Jest zapisywany małymi literami i służy do filtrowania wyników. |
| `--index-json PLIK` | Wymagany plik JSON stanu indeksu, np. `D:\AI\qdrant\rag-index.json`. |
| `-h`, `--help` | Wyświetla pomoc. Tylko ta forma nie wymaga `--index-json`. |

```powershell
rag_qdrant D:\Dokumenty\projekt-a --source project-a --index-json D:\AI\qdrant\rag-index.json
rag_qdrant D:\Notatki --source engineering-notes --index-json D:\AI\qdrant\rag-index.json
```

### Stan kolekcji i źródeł

```text
rag_qdrant --status --index-json PLIK [--json]
rag_qdrant --list-sources --index-json PLIK [--json]
```

| Parametr | Znaczenie |
| --- | --- |
| `--status` | Sprawdza połączenie z Qdrant, zapewnia istnienie kolekcji i pokazuje adres, nazwę, stan oraz liczbę wektorów. Bez `--json` pokazuje także tabelę źródeł. |
| `-l`, `--list-sources` | Pokazuje tagi źródeł, liczbę plików, fragmentów/wektorów i czas ostatniego indeksowania z pliku JSON. |
| `--index-json PLIK` | Wymagany plik JSON stanu indeksu. |
| `--json` | Zwraca dane maszynowe JSON. W głównym trybie CLI jest dozwolony tylko z `--status` albo `--list-sources`. |

`--status` odczytuje stan kolekcji bezpośrednio z Qdrant. Dane o źródłach dla
`--list-sources` pochodzą z lokalnego pliku `--index-json`.

### Wyszukiwanie semantyczne

```text
rag_qdrant search ZAPYTANIE --index-json PLIK [--source TAGI] [--limit LICZBA] --json
```

| Parametr | Znaczenie |
| --- | --- |
| `ZAPYTANIE` | Wymagany tekst wyszukiwany semantycznie. |
| `--index-json PLIK` | Wymagany plik JSON stanu indeksu. Samo wyszukiwanie korzysta z wektorów zapisanych w Qdrant. |
| `-s TAGI`, `--source TAGI` | Opcjonalny tag źródła albo lista tagów rozdzielona przecinkami, np. `project-a,engineering-notes`. Bez parametru przeszukiwana jest cała kolekcja. |
| `--limit LICZBA` | Maksymalna liczba wyników; domyślnie `5`, minimalnie `1`. |
| `--json` | Wymagany. Wynik jest tablicą JSON z oceną podobieństwa, tagiem źródła, ścieżką, nagłówkiem, treścią fragmentu i powiązanymi obrazami. |

```powershell
rag_qdrant search "DMA arbitration" --source project-a,engineering-notes --limit 10 --index-json D:\AI\qdrant\rag-index.json --json
```

## Indeksowanie i hash

Masz rację: przy zwykłym przebiegu każdy znaleziony plik jest ponownie
odczytywany i haszowany SHA-256. Tylko w ten sposób można wykryć zmianę. Cache
nie oszczędza haszowania, ale oszczędza kosztowniejsze dzielenie tekstu,
tworzenie embeddingów i zapis wektorów dla niezmienionych plików.

Przebieg działa tak:

1. Znajduje wszystkie kwalifikujące się pliki `.md` pod `PATH` i oblicza SHA-256 każdego z nich.
2. Plik nieobecny w `--index-json` jest dzielony na fragmenty, otrzymuje embeddingi i jest dodawany do Qdrant.
3. Plik o zmienionym SHA-256 jest aktualizowany: jego stare punkty są usuwane z Qdrant, po czym zapisywane są nowe fragmenty i wektory.
4. Plik o identycznym SHA-256 nie jest dalej przetwarzany ani zapisywany do Qdrant.
5. Wpis JSON dla pliku, którego nie ma już pod bieżącym `PATH`, jest usuwany z cache i Qdrant. Nie dotyczy to plików należących do innego katalogu głównego tego samego źródła.

## Obrazy

Indeksator nie analizuje ani nie opisuje obrazów. Nie odczytuje sidecarów,
nie wykonuje OCR i nie korzysta z usług chmurowych. Ścieżki obrazów powiązanych
z fragmentem Markdown mogą być zapisane jako metadane wyniku, lecz treść obrazu
nie wpływa na embedding ani wyszukiwanie.

## Struktura katalogu

```text
rag-qdrant/
├── README.md
├── requirements.txt
├── bin/                    # uruchamiacze dla PATH
├── rag_qdrant/             # implementacja CLI
└── tests/                  # testy regresji interfejsu
```
