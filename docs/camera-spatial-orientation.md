# Kamera w Orientacji Przestrzennej Robota

## Wprowadzenie

**Orientacja przestrzenna robota** przy użyciu kamery to jeden z fundamentalnych aspektów percepcji robotycznej. Kamery dostarczają robotowi bogatych informacji wizualnych o otoczeniu, umożliwiając mu rozpoznawanie obiektów, szacowanie odległości, planowanie ścieżek i unikanie przeszkód. Robot **Unitree G1 EDU** wyposażony jest w kamerę głębi Intel RealSense D435, która dostarcza zarówno obraz RGB, jak i mapę głębi (depth map), co pozwala na pełną percepcję trójwymiarowej przestrzeni.

W niniejszym artykule przedstawimy, w jaki sposób robot wykorzystuje dane z kamery do orientacji w przestrzeni, jak interpretuje informacje o głębi w obrazie oraz zaprezentujemy praktyczne przykłady przetwarzania obrazu z kamery robota Unitree G1 EDU.

## Kamera Głębi Intel RealSense D435

### Specyfikacja

Robot Unitree G1 EDU wykorzystuje kamerę **Intel RealSense D435**, która łączy w sobie sensor RGB i sensor głębi:

| Parametr | Wartość |
|----------|---------|
| **Sensor RGB** | 1920 × 1080 @ 30 fps |
| **Sensor głębi** | 1280 × 720 @ 90 fps |
| **Zasięg głębi** | 0.1 - 10 m |
| **FOV (pole widzenia)** | 87° × 58° |
| **Technologia głębi** | Stereo IR (aktywna) |
| **Interfejs** | USB 3.0 / USB-C |
| **Wymiary** | 90 × 25 × 25 mm |

### Zasada Działania Kamery Głębi

Kamera RealSense D435 wykorzystuje **stereowizję aktywną** — dwa czujniki podczerwieni (IR) oraz projektor wzorca IR. Proces wyznaczania głębi przebiega następująco:

1. **Projektor IR** emituje niewidoczny wzorzec punktów na scenę
2. **Dwa czujniki IR** (lewy i prawy) rejestrują scenę z niewielkim przesunięciem (baseline)
3. **Algorytm stereo matching** porównuje oba obrazy i wyznacza **dysparycję** (różnicę pozycji tego samego punktu na lewym i prawym obrazie)
4. **Głębia** jest obliczana z dysparycji: im większa dysparycja, tym bliżej znajduje się obiekt

```
Lewy sensor IR ←──── baseline (55mm) ────→ Prawy sensor IR
       \                                        /
        \              Projektor IR             /
         \                 |                   /
          \                ↓                  /
           \         Wzorzec punktów         /
            \           na scenie           /
             ↘                            ↙
              ────────── Obiekt ──────────

Dysparycja (d) = x_L - x_R
Głębia (Z) = (f × B) / d

gdzie: f = ogniskowa, B = baseline, d = dysparycja
```

## Orientacja Robota w Przestrzeni za Pomocą Kamery

### Model Kamery Pinhole

Podstawą orientacji przestrzennej jest zrozumienie modelu kamery. **Model pinhole** opisuje relację między punktem 3D w świecie a jego rzutem 2D na obraz:

```
Punkt 3D (X, Y, Z) → Projekcja → Punkt 2D (u, v)

u = f_x × (X / Z) + c_x
v = f_y × (Y / Z) + c_y

Macierz intrinsic K:
┌           ┐
│ f_x  0  c_x │
│  0  f_y  c_y │
│  0   0    1  │
└           ┘
```

Gdzie:
- **f_x, f_y** — ogniskowa w pikselach (oś X i Y)
- **c_x, c_y** — punkt główny kamery (principal point)
- **X, Y, Z** — współrzędne 3D obiektu w układzie kamery
- **u, v** — współrzędne piksela na obrazie

### Praktyczny Przykład 1: Inicjalizacja Kamery RealSense i Pobranie Obrazu RGB + Głębi

Poniższy kod przedstawia jak połączyć się z kamerą Intel RealSense D435 na robocie Unitree G1 EDU, skonfigurować strumienie RGB i głębi, a następnie pobrać zsynchronizowaną parę ramek. Biblioteka `pyrealsense2` zarządza konfiguracją pipeline, a wyrównanie (alignment) gwarantuje, że piksele obrazu RGB odpowiadają dokładnie pikselom mapy głębi.

```python
import pyrealsense2 as rs
import numpy as np
import cv2

def initialize_realsense_camera():
    """
    Inicjalizacja kamery Intel RealSense D435 na robocie Unitree G1 EDU.
    Konfiguruje strumienie głębi i koloru, włącza wyrównanie ramek.
    """
    # Utworzenie pipeline - zarządza strumieniami danych z kamery
    pipeline = rs.pipeline()
    config = rs.config()

    # Konfiguracja strumieni:
    # - Głębia: 640x480 pikseli, format Z16 (16-bit integer), 30 fps
    # - Kolor: 640x480 pikseli, format BGR8 (kompatybilny z OpenCV), 30 fps
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Uruchomienie pipeline z podaną konfiguracją
    profile = pipeline.start(config)

    # Pobranie sensora głębi i włączenie filtrowania
    depth_sensor = profile.get_device().first_depth_sensor()
    # Skala głębi - konwersja z jednostek sensora na metry
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Skala głębi: {depth_scale} (1 jednostka = {depth_scale} m)")

    # Wyrównanie obrazu głębi do obrazu RGB
    # Dzięki temu każdy piksel (u,v) w obrazie RGB
    # ma odpowiadający mu piksel głębi
    align = rs.align(rs.stream.color)

    return pipeline, align, depth_scale


def capture_aligned_frames(pipeline, align):
    """
    Pobiera zsynchronizowaną parę ramek (RGB + głębia) z kamery.
    Wyrównuje mapę głębi do obrazu kolorowego.
    """
    # Oczekiwanie na kolejną parę ramek
    frames = pipeline.wait_for_frames()

    # Wyrównanie ramek - mapa głębi jest transformowana
    # tak, aby odpowiadała perspektywie kamery RGB
    aligned_frames = align.process(frames)

    # Pobranie wyrównanej ramki głębi i ramki koloru
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None

    # Konwersja do tablic NumPy
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, depth_image


# --- Główne użycie ---
pipeline, align, depth_scale = initialize_realsense_camera()

# Pominięcie pierwszych 30 klatek (auto-ekspozycja kamery)
for _ in range(30):
    pipeline.wait_for_frames()

color_img, depth_img = capture_aligned_frames(pipeline, align)

if color_img is not None:
    # Konwersja mapy głębi na metry
    depth_meters = depth_img * depth_scale

    # Wizualizacja mapy głębi z kolorową skalą
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_img, alpha=0.03),
        cv2.COLORMAP_JET
    )

    # Wyświetlenie obok siebie
    combined = np.hstack((color_img, depth_colormap))
    cv2.imshow("Unitree G1 EDU - RGB | Depth", combined)
    cv2.waitKey(0)

pipeline.stop()
```

**Co robi ten kod:**
- Funkcja `initialize_realsense_camera()` konfiguruje kamerę RealSense D435 — ustawia strumienie głębi (640×480, Z16, 30fps) i koloru (640×480, BGR8, 30fps), uruchamia pipeline oraz tworzy obiekt `align`, który wyrównuje mapę głębi do perspektywy kamery RGB.
- Funkcja `capture_aligned_frames()` pobiera zsynchronizowaną parę ramek i konwertuje je na tablice NumPy, gotowe do przetwarzania przez OpenCV.
- Mapa głębi jest wizualizowana z użyciem kolorowej skali JET, gdzie ciepłe kolory (czerwony) oznaczają obiekty bliskie, a zimne (niebieski) — odległe.

## Mapa Głębi i Chmura Punktów 3D

### Konwersja Pikseli na Współrzędne 3D

Mając obraz RGB, mapę głębi i parametry intrinsic kamery, możemy przekształcić każdy piksel w punkt 3D w przestrzeni. Jest to kluczowe dla orientacji robota — pozwala mu "widzieć" trójwymiarową geometrię otoczenia.

Wzór na konwersję piksela (u, v) z głębią Z na punkt 3D:

```
X = (u - c_x) × Z / f_x
Y = (v - c_y) × Z / f_y
Z = depth(u, v)
```

### Praktyczny Przykład 2: Generowanie Chmury Punktów 3D z Mapy Głębi

Ten kod demonstruje, jak przekształcić mapę głębi z kamery RealSense w trójwymiarową chmurę punktów. Każdy piksel z informacją o głębi jest przeliczany na współrzędne XYZ w układzie kamery, z opcjonalnym przypisaniem koloru z obrazu RGB. Wynikowa chmura punktów może być wizualizowana lub zapisana do pliku PLY.

```python
import pyrealsense2 as rs
import numpy as np
import open3d as o3d

def depth_to_pointcloud(depth_image, color_image, intrinsics, depth_scale):
    """
    Konwersja mapy głębi na chmurę punktów 3D z kolorami.

    Args:
        depth_image: Mapa głębi (H x W), wartości w jednostkach sensora
        color_image: Obraz RGB (H x W x 3)
        intrinsics: Parametry intrinsic kamery (f_x, f_y, c_x, c_y)
        depth_scale: Współczynnik konwersji na metry

    Returns:
        points: Tablica punktów 3D (N x 3) w metrach
        colors: Tablica kolorów (N x 3) znormalizowana [0, 1]
    """
    h, w = depth_image.shape
    f_x, f_y = intrinsics.fx, intrinsics.fy
    c_x, c_y = intrinsics.ppx, intrinsics.ppy

    # Tworzenie siatki współrzędnych pikseli
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Konwersja głębi na metry
    z = depth_image.astype(np.float32) * depth_scale

    # Filtrowanie pikseli bez głębi (z == 0) i zbyt odległych (> 5m)
    valid_mask = (z > 0.1) & (z < 5.0)

    # Obliczenie współrzędnych 3D dla każdego piksela
    x = (u_coords - c_x) * z / f_x
    y = (v_coords - c_y) * z / f_y

    # Zebranie punktów spełniających warunki
    points = np.stack([
        x[valid_mask],
        y[valid_mask],
        z[valid_mask]
    ], axis=-1)

    # Pobranie kolorów odpowiadających punktom (konwersja BGR -> RGB)
    colors = color_image[valid_mask][:, ::-1].astype(np.float32) / 255.0

    return points, colors


def visualize_pointcloud(points, colors):
    """
    Wizualizacja chmury punktów 3D z użyciem Open3D.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Opcjonalnie: estymacja normalnych (przydatna do renderowania)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30
        )
    )

    # Wizualizacja
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Unitree G1 EDU - Chmura Punktów 3D",
        width=1280,
        height=720
    )

    return pcd


# --- Użycie z kamerą RealSense ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# Pobranie parametrów intrinsic kamery
depth_intrinsics = (
    profile.get_stream(rs.stream.depth)
    .as_video_stream_profile()
    .get_intrinsics()
)

align = rs.align(rs.stream.color)

# Pominięcie klatek startowych
for _ in range(30):
    pipeline.wait_for_frames()

frames = align.process(pipeline.wait_for_frames())
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

# Generowanie chmury punktów
points, colors = depth_to_pointcloud(
    depth_image, color_image, depth_intrinsics, depth_scale
)

print(f"Wygenerowano {len(points)} punktów 3D")

# Wizualizacja
pcd = visualize_pointcloud(points, colors)

# Zapis do pliku PLY
o3d.io.write_point_cloud("unitree_g1_scan.ply", pcd)
print("Chmura punktów zapisana do: unitree_g1_scan.ply")

pipeline.stop()
```

**Co robi ten kod:**
- Funkcja `depth_to_pointcloud()` przelicza każdy piksel mapy głębi na punkt 3D w przestrzeni kamery, stosując wzory odwrotnej projekcji (deprojection). Filtruje piksele bez prawidłowej głębi oraz te oddalone ponad 5 metrów.
- Siatka `meshgrid` pozwala na wektoryzowaną (szybką) konwersję wszystkich pikseli jednocześnie zamiast iteracji po każdym pikselu z osobna.
- Funkcja `visualize_pointcloud()` wykorzystuje bibliotekę Open3D do interaktywnej wizualizacji — użytkownik może obracać, przybliżać i eksplorować trójwymiarowe otoczenie robota.
- Wynikowa chmura punktów może być zapisana do pliku PLY i wykorzystana w algorytmach SLAM, planowania ścieżek lub mapowania.

## Wykrywanie Przeszkód na Podstawie Głębi

### Segmentacja Przestrzeni na Podstawie Odległości

Robot musi identyfikować przeszkody w swoim otoczeniu, aby bezpiecznie nawigować. Mapa głębi pozwala na podział przestrzeni na strefy bezpieczeństwa:

```
Strefa krytyczna:    0.0 - 0.5 m   🔴  STOP / Zatrzymanie
Strefa ostrzegawcza: 0.5 - 1.5 m   🟡  Zwolnienie / Omijanie
Strefa bezpieczna:   1.5 - 5.0 m   🟢  Normalna nawigacja
Poza zasięgiem:      > 5.0 m        ⚪  Brak danych
```

### Praktyczny Przykład 3: Detekcja Przeszkód i Wyznaczanie Wolnej Przestrzeni

Ten kod analizuje mapę głębi z kamery RealSense, dzieli pole widzenia na trzy sektory (lewy, środkowy, prawy), a następnie oblicza średnią odległość do przeszkód w każdym sektorze. Na tej podstawie robot podejmuje decyzję nawigacyjną — czy jechać prosto, skręcić, czy się zatrzymać.

```python
import numpy as np
import cv2

class ObstacleDetector:
    """
    Detekcja przeszkód na podstawie mapy głębi z kamery RealSense D435.
    Robot Unitree G1 EDU wykorzystuje ten moduł do bezpiecznej nawigacji.
    """

    # Progi odległości (w metrach)
    CRITICAL_DIST = 0.5    # Strefa krytyczna - natychmiastowe zatrzymanie
    WARNING_DIST = 1.5     # Strefa ostrzegawcza - zwolnienie
    SAFE_DIST = 5.0        # Maksymalny zasięg analizy

    def __init__(self, depth_scale, image_width=640, image_height=480):
        self.depth_scale = depth_scale
        self.width = image_width
        self.height = image_height

        # Definicja sektorów analizy (lewy, środek, prawy)
        # Każdy sektor to 1/3 szerokości obrazu
        third = image_width // 3
        self.sectors = {
            'left':   (0, third),
            'center': (third, 2 * third),
            'right':  (2 * third, image_width)
        }

        # Analizujemy tylko dolną połowę obrazu
        # (bardziej istotna dla przeszkód na drodze robota)
        self.roi_top = image_height // 2
        self.roi_bottom = image_height

    def analyze_depth(self, depth_image):
        """
        Analiza mapy głębi - wyznaczenie odległości w każdym sektorze.

        Returns:
            dict: Odległości i statusy dla każdego sektora
        """
        # Konwersja na metry
        depth_meters = depth_image.astype(np.float32) * self.depth_scale

        # Region zainteresowania - dolna połowa obrazu
        roi = depth_meters[self.roi_top:self.roi_bottom, :]

        results = {}
        for sector_name, (x_start, x_end) in self.sectors.items():
            sector_depth = roi[:, x_start:x_end]

            # Filtrowanie nieprawidłowych wartości
            valid = sector_depth[(sector_depth > 0.1) & (sector_depth < self.SAFE_DIST)]

            if len(valid) > 0:
                min_dist = np.min(valid)
                mean_dist = np.mean(valid)
                # Percentyl 10% - odległość do najbliższych obiektów
                p10_dist = np.percentile(valid, 10)
            else:
                min_dist = mean_dist = p10_dist = float('inf')

            # Klasyfikacja strefy bezpieczeństwa
            if p10_dist < self.CRITICAL_DIST:
                status = 'CRITICAL'
            elif p10_dist < self.WARNING_DIST:
                status = 'WARNING'
            else:
                status = 'SAFE'

            results[sector_name] = {
                'min_distance': min_dist,
                'mean_distance': mean_dist,
                'p10_distance': p10_dist,
                'status': status
            }

        return results

    def get_navigation_command(self, sector_results):
        """
        Na podstawie analizy sektorów generuje komendę nawigacyjną.

        Returns:
            str: Komenda nawigacyjna ('FORWARD', 'TURN_LEFT',
                 'TURN_RIGHT', 'STOP')
        """
        left = sector_results['left']
        center = sector_results['center']
        right = sector_results['right']

        # Jeśli środek jest krytyczny - zatrzymaj się
        if center['status'] == 'CRITICAL':
            return 'STOP'

        # Jeśli środek jest w strefie ostrzegawczej - szukaj wolnej drogi
        if center['status'] == 'WARNING':
            if left['p10_distance'] > right['p10_distance']:
                return 'TURN_LEFT'
            else:
                return 'TURN_RIGHT'

        # Środek bezpieczny - jedź prosto
        return 'FORWARD'

    def visualize(self, color_image, depth_image, sector_results):
        """
        Wizualizacja wyników detekcji przeszkód na obrazie.
        """
        vis = color_image.copy()

        status_colors = {
            'CRITICAL': (0, 0, 255),    # Czerwony
            'WARNING':  (0, 165, 255),   # Pomarańczowy
            'SAFE':     (0, 255, 0)      # Zielony
        }

        for sector_name, (x_start, x_end) in self.sectors.items():
            result = sector_results[sector_name]
            color = status_colors[result['status']]

            # Rysowanie prostokąta sektora
            cv2.rectangle(
                vis,
                (x_start, self.roi_top),
                (x_end, self.roi_bottom),
                color, 2
            )

            # Informacja o odległości
            text = f"{result['p10_distance']:.2f}m"
            cx = (x_start + x_end) // 2 - 30
            cy = self.roi_top + 30
            cv2.putText(vis, text, (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Status
            cv2.putText(vis, result['status'], (cx, cy + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Komenda nawigacyjna
        nav_cmd = self.get_navigation_command(sector_results)
        cv2.putText(vis, f"NAV: {nav_cmd}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return vis


# --- Użycie z kamerą RealSense ---
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

detector = ObstacleDetector(depth_scale)

try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_img = np.asanyarray(depth_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())

        # Analiza przeszkód
        results = detector.analyze_depth(depth_img)
        nav_command = detector.get_navigation_command(results)

        # Wizualizacja
        vis = detector.visualize(color_img, depth_img, results)
        cv2.imshow("Unitree G1 EDU - Obstacle Detection", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
```

**Co robi ten kod:**
- Klasa `ObstacleDetector` dzieli obraz na trzy sektory (lewy, środkowy, prawy) i analizuje dolną połowę obrazu, która odpowiada przestrzeni przed robotem na poziomie podłoża.
- Dla każdego sektora obliczany jest **percentyl 10%** odległości — jest to bardziej stabilna miara niż minimum, która ignoruje pojedyncze szumy w mapie głębi.
- Metoda `get_navigation_command()` implementuje prostą logikę nawigacyjną: jeśli środek jest zablokowany, robot skręca w stronę z większą wolną przestrzenią; jeśli przeszkoda jest zbyt blisko — zatrzymuje się.
- Wizualizacja koloruje sektory odpowiednimi kolorami (czerwony, pomarańczowy, zielony) i wyświetla aktualną komendę nawigacyjną.

## Estymacja Pozycji i Orientacji Kamery

### Visual Odometry

**Odometria wizualna** (Visual Odometry) to technika szacowania ruchu robota na podstawie zmian w kolejnych klatkach obrazu. Robot analizuje punkty charakterystyczne (features) i śledzi ich przesunięcia między klatkami, aby wyznaczyć swoją zmianę pozycji i orientacji.

Proces składa się z następujących kroków:

1. **Detekcja cech** — wykrycie punktów charakterystycznych (np. narożników, krawędzi) w obrazie
2. **Śledzenie cech** — znalezienie tych samych punktów na kolejnej klatce
3. **Estymacja ruchu** — obliczenie macierzy transformacji (rotacja + translacja) z par punktów
4. **Akumulacja** — sumowanie kolejnych transformacji w celu uzyskania pełnej trajektorii

### Praktyczny Przykład 4: Visual Odometry z Użyciem Cech ORB

Poniższy kod implementuje prostą odometrię wizualną opartą na cechach ORB (Oriented FAST and Rotated BRIEF). Algorytm wykrywa punkty kluczowe na kolejnych klatkach z kamery, dopasowuje je, a następnie wyznacza macierz Essential, z której wyciąga rotację i translację kamery. W efekcie robot może śledzić swoją pozycję bez GPS.

```python
import cv2
import numpy as np

class VisualOdometry:
    """
    Odometria wizualna oparta na cechach ORB
    dla robota Unitree G1 EDU.
    """

    def __init__(self, camera_matrix):
        """
        Args:
            camera_matrix: Macierz intrinsic kamery (3x3)
        """
        self.K = camera_matrix
        self.focal = camera_matrix[0, 0]    # Ogniskowa
        self.pp = (                          # Principal point
            camera_matrix[0, 2],
            camera_matrix[1, 2]
        )

        # Detektor ORB - szybki i efektywny
        self.orb = cv2.ORB_create(nfeatures=2000)

        # Matcher cech (Brute-Force z normą Hamminga)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Stan wewnętrzny
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None

        # Akumulowana pozycja i orientacja
        self.position = np.zeros(3)
        self.orientation = np.eye(3)
        self.trajectory = [self.position.copy()]

    def process_frame(self, frame):
        """
        Przetworzenie kolejnej klatki - wyznaczenie ruchu.

        Returns:
            R: Macierz rotacji (3x3) lub None
            t: Wektor translacji (3x1) lub None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detekcja cech ORB
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_frame is None or des is None:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return None, None

        # Dopasowanie cech między klatkami
        matches = self.matcher.match(self.prev_des, des)

        # Sortowanie po odległości (jakości dopasowania)
        matches = sorted(matches, key=lambda m: m.distance)

        # Wybór najlepszych dopasowań (top 70%)
        n_good = int(len(matches) * 0.7)
        good_matches = matches[:max(n_good, 8)]

        if len(good_matches) < 8:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return None, None

        # Ekstrakcja współrzędnych dopasowanych punktów
        pts_prev = np.float32(
            [self.prev_kp[m.queryIdx].pt for m in good_matches]
        )
        pts_curr = np.float32(
            [kp[m.trainIdx].pt for m in good_matches]
        )

        # Wyznaczenie macierzy Essential
        E, mask = cv2.findEssentialMat(
            pts_curr, pts_prev,
            self.focal, self.pp,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return None, None

        # Dekompozycja macierzy Essential na R i t
        _, R, t, mask_pose = cv2.recoverPose(
            E, pts_curr, pts_prev,
            focal=self.focal, pp=self.pp
        )

        # Aktualizacja pozycji i orientacji
        self.position += self.orientation @ t.flatten()
        self.orientation = R @ self.orientation
        self.trajectory.append(self.position.copy())

        # Aktualizacja stanu
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des

        return R, t

    def draw_trajectory(self, canvas_size=600):
        """
        Rysowanie trajektorii robota (widok z góry, płaszczyzna XZ).
        """
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        center = canvas_size // 2

        if len(self.trajectory) < 2:
            return canvas

        # Normalizacja skali do rozmiaru canvas
        traj = np.array(self.trajectory)
        scale = 1.0
        max_range = max(
            np.ptp(traj[:, 0]),
            np.ptp(traj[:, 2])
        )
        if max_range > 0:
            scale = (canvas_size * 0.4) / max_range

        # Rysowanie trajektorii
        for i in range(1, len(self.trajectory)):
            pt1 = (
                int(center + self.trajectory[i-1][0] * scale),
                int(center + self.trajectory[i-1][2] * scale)
            )
            pt2 = (
                int(center + self.trajectory[i][0] * scale),
                int(center + self.trajectory[i][2] * scale)
            )
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)

        # Oznaczenie aktualnej pozycji
        curr = (
            int(center + self.position[0] * scale),
            int(center + self.position[2] * scale)
        )
        cv2.circle(canvas, curr, 5, (0, 0, 255), -1)

        # Etykieta
        cv2.putText(canvas, "Trajektoria Unitree G1 EDU",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 2)

        return canvas


# --- Użycie ---
# Macierz intrinsic kamery RealSense D435 (przykładowe wartości)
camera_matrix = np.array([
    [615.0,   0.0, 320.0],
    [  0.0, 615.0, 240.0],
    [  0.0,   0.0,   1.0]
])

vo = VisualOdometry(camera_matrix)

cap = cv2.VideoCapture(0)  # lub strumień RealSense

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    R, t = vo.process_frame(frame)

    # Wizualizacja trajektorii
    traj_img = vo.draw_trajectory()

    cv2.imshow("Obraz z kamery", frame)
    cv2.imshow("Trajektoria", traj_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Co robi ten kod:**
- Klasa `VisualOdometry` implementuje monokularkową odometrię wizualną z użyciem detektora **ORB** (szybka alternatywa dla SIFT/SURF), który wykrywa i opisuje punkty charakterystyczne na każdej klatce.
- Dopasowanie cech między kolejnymi klatkami odbywa się za pomocą **BFMatcher** (Brute-Force) z metryką Hamminga, optymalną dla binarnych deskryptorów ORB.
- Z dopasowanych par punktów wyznaczana jest **macierz Essential** (metodą RANSAC), a następnie rozkładana na macierz rotacji **R** i wektor translacji **t** za pomocą `cv2.recoverPose()`.
- Akumulacja kolejnych transformacji daje pełną trajektorię robota, wizualizowaną jako widok z góry (płaszczyzna XZ).

## Rozpoznawanie Sceny i Segmentacja Semantyczna

### Segmentacja Semantyczna z Głębią

**Segmentacja semantyczna** przypisuje każdemu pikselowi obrazu etykietę klasy (np. podłoga, ściana, krzesło, osoba). W połączeniu z mapą głębi robot może zbudować trójwymiarową mapę semantyczną otoczenia — wie nie tylko *co* widzi, ale także *gdzie* to jest w przestrzeni.

### Praktyczny Przykład 5: Segmentacja Semantyczna z Mapą Głębi

Poniższy przykład łączy segmentację semantyczną (z modelem DeepLabV3+) z mapą głębi z kamery RealSense. Dla każdego segmentu (np. „osoba", „krzesło") obliczana jest średnia odległość, co daje robotowi kontekstową informację o otoczeniu — np. „osoba na wprost w odległości 2.3 m".

```python
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
import numpy as np
import cv2

class SemanticDepthMapper:
    """
    Segmentacja semantyczna połączona z mapą głębi.
    Buduje 3D mapę semantyczną otoczenia robota Unitree G1 EDU.
    """

    # Klasy modelu DeepLabV3 (COCO subset)
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    # Kolory dla wizualizacji (BGR)
    COLORS = np.random.RandomState(42).randint(
        0, 255, (len(CLASSES), 3)
    ).tolist()

    def __init__(self):
        # Model segmentacji semantycznej
        self.model = deeplabv3_resnet101(pretrained=True)
        self.model.eval()

        # Transformacja wejściowa
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(520),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def segment(self, color_image):
        """
        Segmentacja semantyczna obrazu.

        Returns:
            labels: Mapa etykiet (H x W), każdy piksel = ID klasy
        """
        # Konwersja BGR -> RGB
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]

        # Mapa etykiet - argmax po klasach
        labels = output.argmax(0).cpu().numpy().astype(np.uint8)

        # Skalowanie do rozmiaru oryginalnego obrazu
        labels = cv2.resize(
            labels,
            (color_image.shape[1], color_image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        return labels

    def analyze_scene(self, labels, depth_image, depth_scale):
        """
        Analiza sceny - powiązanie segmentów z głębią.

        Returns:
            list: Wykryte obiekty z informacją o odległości
        """
        depth_m = depth_image.astype(np.float32) * depth_scale
        detected_objects = []

        for class_id in range(1, len(self.CLASSES)):  # Pomijamy 'background'
            mask = labels == class_id

            if mask.sum() < 500:  # Minimalna liczba pikseli
                continue

            # Odległość do obiektu
            object_depth = depth_m[mask]
            valid_depth = object_depth[
                (object_depth > 0.1) & (object_depth < 10.0)
            ]

            if len(valid_depth) == 0:
                continue

            # Bounding box segmentu
            ys, xs = np.where(mask)
            bbox = (xs.min(), ys.min(), xs.max(), ys.max())

            # Środek obiektu
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            detected_objects.append({
                'class': self.CLASSES[class_id],
                'class_id': class_id,
                'distance': float(np.median(valid_depth)),
                'bbox': bbox,
                'center': (center_x, center_y),
                'pixel_count': int(mask.sum())
            })

        # Sortowanie po odległości (najbliższe najpierw)
        detected_objects.sort(key=lambda x: x['distance'])
        return detected_objects

    def visualize(self, color_image, labels, detected_objects):
        """
        Wizualizacja segmentacji z informacją o odległościach.
        """
        vis = color_image.copy()

        # Nałożenie maski segmentacji (półprzezroczysta)
        overlay = np.zeros_like(color_image)
        for class_id in range(1, len(self.CLASSES)):
            mask = labels == class_id
            if mask.sum() > 0:
                overlay[mask] = self.COLORS[class_id]

        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        # Etykiety z odległościami
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['bbox']
            cx, cy = obj['center']

            # Ramka
            color = self.COLORS[obj['class_id']]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Etykieta: klasa + odległość
            label = f"{obj['class']} ({obj['distance']:.1f}m)"
            cv2.putText(vis, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return vis


# --- Użycie ---
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

mapper = SemanticDepthMapper()

for _ in range(30):
    pipeline.wait_for_frames()

frames = align.process(pipeline.wait_for_frames())
depth_img = np.asanyarray(frames.get_depth_frame().get_data())
color_img = np.asanyarray(frames.get_color_frame().get_data())

# Segmentacja
labels = mapper.segment(color_img)

# Analiza sceny z głębią
objects = mapper.analyze_scene(labels, depth_img, depth_scale)

# Wypisanie wykrytych obiektów
print("Wykryte obiekty w otoczeniu robota:")
for obj in objects:
    print(f"  {obj['class']:15s} - odległość: {obj['distance']:.2f} m")

# Wizualizacja
vis = mapper.visualize(color_img, labels, objects)
cv2.imshow("Unitree G1 EDU - Semantic Depth Map", vis)
cv2.waitKey(0)

pipeline.stop()
```

**Co robi ten kod:**
- Klasa `SemanticDepthMapper` łączy model segmentacji semantycznej **DeepLabV3+** (wytrenowany na zbiorze COCO) z mapą głębi z kamery RealSense.
- Metoda `segment()` przetwarza obraz RGB przez sieć neuronową, uzyskując mapę etykiet — każdy piksel ma przypisaną klasę (osoba, krzesło, stół itd.).
- Metoda `analyze_scene()` dla każdego wykrytego segmentu oblicza **medianę głębi** (bardziej odporna na szumy niż średnia), bounding box i środek obiektu.
- Wynikowa wizualizacja nakłada półprzezroczystą maskę kolorów na obraz oraz etykiety z odległością do każdego obiektu — robot wie, że np. „osoba" jest 2.3 m na wprost, a „krzesło" 1.1 m po lewej.

## Estymacja Płaszczyzny Podłogi

### Praktyczny Przykład 6: Detekcja Płaszczyzny Podłogi z Chmury Punktów

Robot musi znać położenie podłogi, aby prawidłowo planować kroki i unikać nierówności. Ten przykład wykorzystuje algorytm **RANSAC** na chmurze punktów 3D do znalezienia dominującej płaszczyzny (podłogi). Pozwala to robotowi zrozumieć geometrię terenu — nachylenie, wysokość i granice podłogi.

```python
import numpy as np
import open3d as o3d

class FloorDetector:
    """
    Detekcja płaszczyzny podłogi z chmury punktów 3D.
    Wykorzystuje RANSAC do estymacji równania płaszczyzny.
    """

    def __init__(self, distance_threshold=0.02, ransac_n=3,
                 num_iterations=1000):
        """
        Args:
            distance_threshold: Maksymalna odległość punktu od płaszczyzny
                                aby został uznany za inlier (w metrach)
            ransac_n: Liczba punktów do estymacji modelu
            num_iterations: Liczba iteracji RANSAC
        """
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations

    def detect_floor(self, points, colors=None):
        """
        Detekcja płaszczyzny podłogi z chmury punktów.

        Args:
            points: Tablica punktów 3D (N x 3)
            colors: Tablica kolorów (N x 3), opcjonalnie

        Returns:
            plane_model: Równanie płaszczyzny [a, b, c, d]
                         gdzie ax + by + cz + d = 0
            floor_points: Punkty należące do podłogi
            obstacle_points: Punkty powyżej podłogi (przeszkody)
        """
        # Utworzenie chmury punktów Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # RANSAC - znalezienie dominującej płaszczyzny
        plane_model, inlier_indices = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations
        )

        a, b, c, d = plane_model
        print(f"Równanie płaszczyzny podłogi: "
              f"{a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

        # Wektor normalny płaszczyzny
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        # Kąt nachylenia podłogi (od pionu)
        up_vector = np.array([0, -1, 0])  # Y skierowane w dół w układzie kamery
        angle_rad = np.arccos(
            np.clip(np.dot(normal, up_vector), -1.0, 1.0)
        )
        angle_deg = np.degrees(angle_rad)
        print(f"Nachylenie podłogi: {angle_deg:.1f}°")

        # Podział na podłogę i przeszkody
        all_indices = set(range(len(points)))
        inlier_set = set(inlier_indices)
        outlier_indices = list(all_indices - inlier_set)

        floor_points = points[inlier_indices]
        obstacle_points = points[outlier_indices]

        floor_colors = colors[inlier_indices] if colors is not None else None
        obstacle_colors = colors[outlier_indices] if colors is not None else None

        return {
            'plane_model': plane_model,
            'normal': normal,
            'angle_deg': angle_deg,
            'floor_points': floor_points,
            'floor_colors': floor_colors,
            'obstacle_points': obstacle_points,
            'obstacle_colors': obstacle_colors,
            'floor_height': -d / c if abs(c) > 1e-6 else 0
        }

    def visualize(self, result):
        """
        Wizualizacja: podłoga na zielono, przeszkody na czerwono.
        """
        # Chmura podłogi - zielona
        floor_pcd = o3d.geometry.PointCloud()
        floor_pcd.points = o3d.utility.Vector3dVector(result['floor_points'])
        floor_pcd.paint_uniform_color([0.0, 0.8, 0.0])

        # Chmura przeszkód - czerwona
        obstacle_pcd = o3d.geometry.PointCloud()
        obstacle_pcd.points = o3d.utility.Vector3dVector(
            result['obstacle_points']
        )
        obstacle_pcd.paint_uniform_color([0.8, 0.0, 0.0])

        # Wizualizacja
        o3d.visualization.draw_geometries(
            [floor_pcd, obstacle_pcd],
            window_name="Unitree G1 EDU - Floor Detection",
            width=1280, height=720
        )


# --- Użycie z danymi z kamery RealSense ---
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
intrinsics = (
    profile.get_stream(rs.stream.depth)
    .as_video_stream_profile()
    .get_intrinsics()
)

align = rs.align(rs.stream.color)

for _ in range(30):
    pipeline.wait_for_frames()

frames = align.process(pipeline.wait_for_frames())
depth_img = np.asanyarray(frames.get_depth_frame().get_data())
color_img = np.asanyarray(frames.get_color_frame().get_data())

# Generowanie chmury punktów (z wcześniejszego przykładu)
h, w = depth_img.shape
u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
z = depth_img.astype(np.float32) * depth_scale
valid = (z > 0.1) & (z < 5.0)

x = (u_coords - intrinsics.ppx) * z / intrinsics.fx
y = (v_coords - intrinsics.ppy) * z / intrinsics.fy

points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
colors = color_img[valid][:, ::-1].astype(np.float32) / 255.0

# Detekcja podłogi
detector = FloorDetector(distance_threshold=0.02)
result = detector.detect_floor(points, colors)

print(f"\nPunkty podłogi: {len(result['floor_points'])}")
print(f"Punkty przeszkód: {len(result['obstacle_points'])}")
print(f"Wysokość podłogi: {result['floor_height']:.3f} m")

# Wizualizacja
detector.visualize(result)

pipeline.stop()
```

**Co robi ten kod:**
- Klasa `FloorDetector` stosuje algorytm **RANSAC** (Random Sample Consensus) na chmurze punktów 3D, aby znaleźć dominującą płaszczyznę — podłogę.
- RANSAC losowo wybiera 3 punkty, wyznacza z nich płaszczyznę, a następnie liczy ile punktów leży blisko tej płaszczyzny (inliers). Po wielu iteracjach wybiera płaszczyznę z największą liczbą inlierów.
- Z równania płaszczyzny (ax + by + cz + d = 0) robot wylicza **kąt nachylenia** podłogi względem pionu — jeśli kąt jest bliski 0°, podłoga jest pozioma; większy kąt oznacza pochyłość.
- Wizualizacja koloruje punkty podłogi na zielono, a przeszkody (wszystko powyżej podłogi) na czerwono, dając intuicyjny obraz terenu.

## Śledzenie Obiektów w Czasie Rzeczywistym

### Praktyczny Przykład 7: Śledzenie Obiektu z Estymacją Odległości

W tym przykładzie robot śledzi wybrany obiekt w sekwencji wideo z kamery RealSense, jednocześnie monitorując jego odległość. Tracker CSRT z OpenCV zapewnia precyzyjne śledzenie, a mapa głębi dostarcza ciągłą informację o tym, jak daleko jest śledzony obiekt. Jest to przydatne np. do podążania za osobą.

```python
import cv2
import numpy as np
import pyrealsense2 as rs

class ObjectTracker3D:
    """
    Śledzenie obiektu w 3D z użyciem trackera OpenCV
    i mapy głębi z kamery RealSense D435.
    """

    def __init__(self):
        # Inicjalizacja kamery RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = self.pipeline.start(config)
        self.depth_scale = (
            profile.get_device().first_depth_sensor().get_depth_scale()
        )
        self.align = rs.align(rs.stream.color)

        # Pobranie parametrów intrinsic
        self.intrinsics = (
            profile.get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        # Tracker (CSRT - dokładny, KCF - szybki)
        self.tracker = None
        self.tracking = False
        self.bbox = None

        # Historia pozycji 3D obiektu
        self.position_history = []

    def get_frames(self):
        """Pobranie wyrównanej pary ramek."""
        frames = self.align.process(self.pipeline.wait_for_frames())
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        color = np.asanyarray(frames.get_color_frame().get_data())
        return color, depth

    def get_object_3d_position(self, depth_image, bbox):
        """
        Obliczenie pozycji 3D obiektu na podstawie bounding box i głębi.

        Args:
            depth_image: Mapa głębi
            bbox: (x, y, w, h) - bounding box obiektu

        Returns:
            (X, Y, Z): Pozycja 3D w metrach lub None
        """
        x, y, w, h = [int(v) for v in bbox]

        # Region głębi odpowiadający obiektowi
        # Używamy centralnych 60% bbox (unikamy krawędzi)
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        roi = depth_image[
            y + margin_y : y + h - margin_y,
            x + margin_x : x + w - margin_x
        ]

        if roi.size == 0:
            return None

        # Konwersja na metry i filtrowanie
        roi_meters = roi.astype(np.float32) * self.depth_scale
        valid = roi_meters[(roi_meters > 0.1) & (roi_meters < 10.0)]

        if len(valid) == 0:
            return None

        Z = float(np.median(valid))

        # Środek obiektu w pikselach
        cx = x + w / 2
        cy = y + h / 2

        # Konwersja na współrzędne 3D
        X = (cx - self.intrinsics.ppx) * Z / self.intrinsics.fx
        Y = (cy - self.intrinsics.ppy) * Z / self.intrinsics.fy

        return (X, Y, Z)

    def run(self):
        """
        Główna pętla śledzenia.
        Naciśnij 's' aby wybrać obiekt, 'q' aby zakończyć.
        """
        print("Naciśnij 's' aby wybrać obiekt do śledzenia")
        print("Naciśnij 'q' aby zakończyć")

        # Pominięcie klatek startowych
        for _ in range(30):
            self.pipeline.wait_for_frames()

        try:
            while True:
                color, depth = self.get_frames()
                vis = color.copy()

                if self.tracking and self.tracker is not None:
                    # Aktualizacja trackera
                    success, self.bbox = self.tracker.update(color)

                    if success:
                        x, y, w, h = [int(v) for v in self.bbox]

                        # Pozycja 3D
                        pos_3d = self.get_object_3d_position(depth, self.bbox)

                        if pos_3d is not None:
                            X, Y, Z = pos_3d
                            self.position_history.append(pos_3d)

                            # Rysowanie bounding box
                            cv2.rectangle(
                                vis, (x, y), (x + w, y + h),
                                (0, 255, 0), 2
                            )

                            # Informacje o pozycji 3D
                            info = [
                                f"X: {X:.2f}m",
                                f"Y: {Y:.2f}m",
                                f"Z: {Z:.2f}m (odległość)"
                            ]
                            for i, text in enumerate(info):
                                cv2.putText(
                                    vis, text,
                                    (x, y - 10 - i * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 2
                                )

                            # Celownik w środku obiektu
                            cx, cy = x + w // 2, y + h // 2
                            cv2.drawMarker(
                                vis, (cx, cy),
                                (0, 0, 255),
                                cv2.MARKER_CROSS, 20, 2
                            )
                    else:
                        self.tracking = False
                        cv2.putText(
                            vis, "Obiekt utracony!",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 2
                        )

                cv2.imshow("Unitree G1 EDU - 3D Object Tracking", vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    # Wybór obiektu do śledzenia
                    bbox = cv2.selectROI(
                        "Unitree G1 EDU - 3D Object Tracking",
                        color, fromCenter=False
                    )
                    if bbox[2] > 0 and bbox[3] > 0:
                        self.tracker = cv2.TrackerCSRT_create()
                        self.tracker.init(color, bbox)
                        self.bbox = bbox
                        self.tracking = True
                elif key == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


# --- Uruchomienie ---
tracker = ObjectTracker3D()
tracker.run()
```

**Co robi ten kod:**
- Klasa `ObjectTracker3D` łączy tracker wizualny **CSRT** (Channel and Spatial Reliability Tracker) z mapą głębi, umożliwiając śledzenie obiektu w trzech wymiarach.
- Po naciśnięciu klawisza `s` użytkownik zaznacza interesujący obiekt na obrazie (bounding box). Tracker automatycznie śledzi ten obiekt w kolejnych klatkach.
- Metoda `get_object_3d_position()` oblicza pozycję 3D (X, Y, Z) śledzonego obiektu, analizując głębię w centralnych 60% bounding boxa (marginesy eliminują tło w tle obiektu).
- Na ekranie wyświetlane są współrzędne 3D obiektu, co pozwala robotowi np. podążać za osobą utrzymując określoną odległość.

## Filtrowanie i Poprawa Jakości Mapy Głębi

### Praktyczny Przykład 8: Filtrowanie Mapy Głębi

Surowa mapa głębi z kamery RealSense zawiera szumy i artefakty (szczególnie na krawędziach obiektów i powierzchniach odbijających). Poniższy kod stosuje zestaw filtrów sprzętowych i programowych, aby uzyskać gładszą, bardziej wiarygodną mapę głębi, co poprawia precyzję wszystkich algorytmów bazujących na głębi.

```python
import pyrealsense2 as rs
import numpy as np
import cv2

class DepthFilter:
    """
    Filtrowanie i poprawa jakości mapy głębi
    z kamery Intel RealSense D435 na robocie Unitree G1 EDU.
    """

    def __init__(self):
        # Filtry RealSense (przetwarzanie sprzętowe/firmware)

        # Filtr decymacji - zmniejsza rozdzielczość (szybsze przetwarzanie)
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 2)

        # Filtr przestrzenny - wygładza mapę, zachowując krawędzie
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.spatial.set_option(rs.option.holes_fill, 1)

        # Filtr czasowy - stabilizuje mapę między klatkami
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)

        # Filtr wypełniania dziur
        self.hole_filling = rs.hole_filling_filter()

    def filter_realsense(self, depth_frame):
        """
        Filtrowanie ramki głębi z użyciem filtrów RealSense.
        Kolejność filtrów ma znaczenie!
        """
        # 1. Decymacja (redukcja rozdzielczości)
        filtered = self.decimation.process(depth_frame)

        # 2. Filtr przestrzenny (wygładzanie)
        filtered = self.spatial.process(filtered)

        # 3. Filtr czasowy (stabilizacja między klatkami)
        filtered = self.temporal.process(filtered)

        # 4. Wypełnianie dziur
        filtered = self.hole_filling.process(filtered)

        return filtered

    @staticmethod
    def bilateral_filter(depth_image, d=5, sigma_color=75, sigma_space=75):
        """
        Filtr bilateralny - wygładza zachowując krawędzie.
        Stosowany na mapie głębi jako tablica NumPy.
        """
        # Normalizacja do 8-bit dla filtra bilateralnego
        depth_normalized = cv2.normalize(
            depth_image, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        filtered = cv2.bilateralFilter(
            depth_normalized, d, sigma_color, sigma_space
        )

        # Przywrócenie oryginalnej skali
        scale = depth_image.max() / 255.0 if depth_image.max() > 0 else 1.0
        return (filtered.astype(np.float32) * scale).astype(np.uint16)

    @staticmethod
    def inpainting(depth_image, max_depth=10000):
        """
        Wypełnianie brakujących wartości głębi (dziur)
        za pomocą interpolacji OpenCV.
        """
        # Maska dziur (piksele z głębią = 0)
        mask = (depth_image == 0).astype(np.uint8)

        if mask.sum() == 0:
            return depth_image

        # Normalizacja do 8-bit
        depth_8bit = cv2.normalize(
            depth_image, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Inpainting (metoda Telea)
        inpainted = cv2.inpaint(depth_8bit, mask, 5, cv2.INPAINT_TELEA)

        # Przywrócenie skali
        scale = depth_image.max() / 255.0 if depth_image.max() > 0 else 1.0
        result = (inpainted.astype(np.float32) * scale).astype(np.uint16)

        return result


# --- Porównanie surowej i przefiltrowanej mapy głębi ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

align = rs.align(rs.stream.color)
depth_filter = DepthFilter()

for _ in range(30):
    pipeline.wait_for_frames()

try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Surowa mapa głębi
        raw_depth = np.asanyarray(depth_frame.get_data())

        # Filtrowanie sprzętowe RealSense
        filtered_frame = depth_filter.filter_realsense(depth_frame)
        hw_filtered = np.asanyarray(filtered_frame.get_data())

        # Dodatkowe filtrowanie programowe (bilateralne)
        sw_filtered = DepthFilter.bilateral_filter(hw_filtered)

        # Wizualizacja porównawcza
        raw_color = cv2.applyColorMap(
            cv2.convertScaleAbs(raw_depth, alpha=0.03),
            cv2.COLORMAP_JET
        )
        hw_color = cv2.applyColorMap(
            cv2.convertScaleAbs(hw_filtered, alpha=0.03),
            cv2.COLORMAP_JET
        )
        sw_color = cv2.applyColorMap(
            cv2.convertScaleAbs(sw_filtered, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Etykiety
        cv2.putText(raw_color, "Surowa", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(hw_color, "Filtr HW", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(sw_color, "Filtr HW+SW", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        combined = np.hstack((raw_color, hw_color, sw_color))
        cv2.imshow("Unitree G1 EDU - Depth Filtering", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
```

**Co robi ten kod:**
- Klasa `DepthFilter` implementuje kaskadę filtrów poprawiających jakość mapy głębi. Filtry sprzętowe RealSense (decymacja, przestrzenny, czasowy, wypełnianie dziur) działają bezpośrednio na danych z sensora.
- **Filtr decymacji** zmniejsza rozdzielczość mapy głębi (mniejszy szum, szybsze przetwarzanie). **Filtr przestrzenny** wygładza mapę zachowując krawędzie obiektów. **Filtr czasowy** uśrednia dane między kolejnymi klatkami, eliminując migotanie.
- Dodatkowy **filtr bilateralny** (OpenCV) dalej wygładza mapę, zachowując ostre krawędzie między obiektami.
- Wizualizacja porównawcza pokazuje obok siebie mapę surową, po filtrach sprzętowych i po pełnym przetwarzaniu — różnica jest szczególnie widoczna na krawędziach obiektów i jednolitych powierzchniach.

## Podsumowanie

Kamera głębi jest jednym z najważniejszych sensorów robota humanoidalnego. Połączenie obrazu RGB z mapą głębi daje robotowi Unitree G1 EDU możliwość:

- **Percepcji 3D** — tworzenia chmur punktów i trójwymiarowych modeli otoczenia
- **Nawigacji** — wykrywania przeszkód i planowania bezpiecznych ścieżek
- **Lokalizacji** — szacowania własnej pozycji i orientacji w przestrzeni (odometria wizualna)
- **Rozpoznawania** — identyfikacji obiektów z informacją o ich odległości
- **Analizy terenu** — detekcji płaszczyzny podłogi i nachylenia powierzchni
- **Śledzenia** — ciągłego monitorowania pozycji 3D wybranych obiektów
- **Filtrowania** — poprawy jakości danych sensorycznych dla wyższej precyzji

Każdy z przedstawionych przykładów może być rozbudowany i zintegrowany z systemem ROS2 robota Unitree G1 EDU, tworząc pełny pipeline percepcji wizualnej wspierający autonomiczną nawigację i interakcję z otoczeniem.
