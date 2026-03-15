# Proponowane zadania po przeglądzie kodu

## 1) Zadanie: poprawa literówki
**Problem:** W komentarzu koloru dla QR użyto zapisu „blue-ish”.

**Proponowane zadanie:**
- Zmienić komentarz `# blue-ish` na `# bluish` (lub spójny opis po polsku/angielsku), żeby usunąć literówkę/stylistyczny zgrzyt i ujednolicić język komentarzy.

## 2) Zadanie: usunięcie błędu
**Problem:** Import samego pakietu (np. `from robo_eye_sense.results import Detection`) przechodzi przez `robo_eye_sense/__init__.py`, które od razu importuje `RoboEyeDetector`, a to natychmiast importuje `cv2`. W środowiskach bez bibliotek GUI OpenCV (np. brak `libGL.so.1`) powoduje to błąd już na etapie importu modułów niezależnych od kamery/GUI.

**Proponowane zadanie:**
- Wprowadzić lazy import w `robo_eye_sense/__init__.py` (np. przez `__getattr__`), tak aby `RoboEyeDetector` był importowany dopiero przy faktycznym użyciu.
- Alternatywnie ograniczyć importy w `__init__.py` tylko do lekkich modułów (`results`) i zostawić `RoboEyeDetector` do importu bezpośredniego.

## 3) Zadanie: korekta komentarza / rozbieżności dokumentacji
**Problem:** README opisuje wariant headless („zamień `opencv-python` na `opencv-python-headless`”), ale `requirements.txt` i `pyproject.toml` deklarują twardo `opencv-python`, co utrudnia środowiska CI/headless i jest niespójne z rekomendacją.

**Proponowane zadanie:**
- Ujednolicić dokumentację i zależności: albo dodać extras (np. `gui`/`headless`) i odzwierciedlić je w README, albo doprecyzować, że użytkownik musi ręcznie podmienić zależność przed instalacją.

## 4) Zadanie: ulepszenie testu
**Problem:** Brakuje testu regresyjnego, który pilnuje, że import lekkich modułów pakietu nie wymaga `cv2`/`libGL`.

**Proponowane zadanie:**
- Dodać test uruchamiany w osobnym procesie (subprocess), który sprawdza, że import `robo_eye_sense.results` działa bez importowania `cv2`.
- Po wdrożeniu lazy importu dodać drugi test, że `from robo_eye_sense import RoboEyeDetector` działa poprawnie, gdy `cv2` jest dostępne.
