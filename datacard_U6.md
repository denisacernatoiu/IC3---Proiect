# Data Card — UrbanTwin: NO₂ la sol București (S5P + OpenAQ)

## 1. Scop
Estimarea concentrației de NO₂ la sol (µg/m³), măsurată de stațiile din București, pornind de la coloana troposferică de NO₂ observată de satelitul Sentinel-5P.

- **Input (feature):** `no2_column_mol_m2` (media zilnică S5P peste București)
- **Target:** `no2_ug_m3` (media zilnică a unui senzor OpenAQ)

## 2. Surse

| | Sentinel-5P (S5P) | OpenAQ |
|---|---|---|
| Sursă | Google Earth Engine, `COPERNICUS/S5P/OFFL/L3_NO2` | OpenAQ API v3, `/sensors/{id}/days` |
| Variabilă | `tropospheric_NO2_column_number_density` | NO₂, medie zilnică |
| Unitate | mol/m² | µg/m³ |
| Acoperire spațială | dreptunghi 25.95–26.25°E, 44.33–44.55°N | 7 stații (B-1…B-7), 14 senzori (2 per stație) |
| Rezoluție spațială | ~1113 m (scale în `reduceRegion`), mediat pe tot dreptunghiul | punctual (la stație) |
| Rezoluție temporală | zilnică (media imaginilor OFFL din zi) | zilnică (ora locală Europe/Bucharest) |
| Perioadă | 2024-01-01 → 2026-10-25 | 2024-01-01 → 2026-10-25 |
| Scripturi | `get_s5p_data.js` (rulat manual în GEE Code Editor, export CSV) | `get_openaq_data.py` |

## 3. Dimensiuni și goluri

| Set | Rânduri | Observații |
|---|---|---|
| S5P brut | 1029 zile | **745 valide**, **284 lipsă (~27.6%)** — zile fără imagini utile (nori, zăpadă, calitate scăzută) |
| OpenAQ brut | 11 998 rânduri senzor-zi | fără valori lipsă după filtrare |
| Combinat| **8730 observații** | **710 zile distincte** |

- Zilele fără valoare S5P sunt eliminate

## 4. Preprocesare
1. Conversie la `datetime`, normalizare la zi, conversie numerică.
2. Eliminare zile S5P fără valoare.
3. Medie S5P pe zi, apoi `merge` many-to-one pe `date` cu OpenAQ.
4. Sortare după `date`, `sensor_id`.

## 5. Împărțire train / val / test
- **Cronologică, pe date unice**  
    70% / 15% / 15% 
- Toți senzorii din aceeași zi ajung în același set, deci nu există scurgere de informație între seturi prin aceeași valoare S5P.
- Număr de observații: 
    * Train = 5875, 
    * Val = 1494, 
    * Test = 1361.
- DataLoader: batch 32, `shuffle=True` doar pe train.


## 5. Citare
- Copernicus Sentinel-5P (processed by ESA), TROPOMI Level 2 Nitrogen Dioxide, via Google Earth Engine.
- OpenAQ, https://openaq.org