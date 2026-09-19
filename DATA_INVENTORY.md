# DATA INVENTORY

## A. Dataset Card

- **Total files**: 38 files — types: .csv(3), .doe(3), .m(25), .mat(1), .pdf(3), .txt(1), .xls(1), .zip(1)
- **Image data**: No
- **README / data-dictionary**: Yes — `Readme.txt`
- **PDFs**: `Model 1.pdf`, `Model 2.pdf`, `Model 3.pdf` (process-model diagrams, not machine-readable text)

**Readme.txt (first 60 lines):**
```
_______Short description of the files contained in this dataset_______



"""""""""""""""""/Model 1/"""""""""""""""""
"""The first model has 10 features presented in the dataset 

Model 1.doe is a Rockwell Arena Simulation model, it was designed in version 15 of the software
Model 1.pdf is a process model, with description of the logic of the simulation model, could be used to redo the model in any other software
Model 1.csv resulting dataset file, was used to run an experiment. 



"""""""""""""""""/Model 2/"""""""""""""""""
"""The second dataset contains 16 features (including metadata)
 
Model 2.doe is a Rockwell Arena Simulation model, it was designed in version 15 of the software
Model 2.pdf is a process model, with description of the logic of the simulation model, could be used to redo the model in any other software
Model 2.csv resulting dataset file, was used to run an experiment



"""""""""""""""""/Model 3/"""""""""""""""""
"""Compared to the first two datasets,  the third contains more complex data with 77 features. The first group of features, utilization, shows the level of usage of a particular resource and the second group, queue, contains all features representing number of parts waiting to be processed. The group part counters include counters by SKU type at the assembly cells and total number of products produced after a quality check
 
Model 3.doe is a Rockwell Arena Simulation model, it was designed in version 15 of the software
Model 3.pdf is a process model, with description of the logic of the simulation model, could be used to redo the model in any other software
Model 3.csv resulting dataset file, was used to run an experiment
ParametersFile.xls contains settings for the simulation model, must be connected to the .doe file using Arena Simulation software in order for model to run correctly
```

**Column keyword scan:**

  - **product**: YES — e.g. `c_TotalProducts`
  - **quality**: YES — e.g. `Quality_Queue`, `Quality_Util`
  - **queue**: YES — e.g. `Cell2_Queue`, `Quality_Queue`, `Milling Queue Time`, `Blanking_SKU3_Queue`, `Press4_Queue`
  - **sku**: YES — e.g. `SKU3_Transport_Time`, `c_Cell1__SKU4`, `SKU2_NVA_Time`, `SKU1_Other_Time`, `c_Cell4__SKU4`
  - **time**: YES — e.g. `SKU3_Transport_Time`, `Milling Queue Time`, `SKU2_NVA_Time`, `SKU1_Other_Time`, `SKU1_NVA_Time`
  - **util**: YES — e.g. `Forklift_Util`, `Paint2_Util`, `Drilling Utilization`, `Cell4_Util`, `Press2_Util`
  - **utilization**: YES — e.g. `Drilling Utilization`, `Assembly Utilization`, `Milling Utilization`
  - **wait**: YES — e.g. `SKU2_Wait_Time`, `Assembly Waiting Time`, `SKU1_Wait_Time`, `Drilling Waiting Time`, `SKU3_Wait_Time`
  - **waiting**: YES — e.g. `Drilling Waiting Time`, `Milling Waiting Time`, `Assembly Waiting Time`
  - **Not found**: batch, breakdown, changeover, cost, defect, downtime, facility, lot, price, profit, revenue, rework, scrap, station, timestamp, yield

**What one row likely represents:**
  - Model 1 & 2: one simulation *run* (experiment replicate) — GUESS
  - Model 3: one *time snapshot* within a long simulation (Time_Now column present) — GUESS

**Identifiers that could join files:**
  - No shared key column across model files; they are independent simulation experiments. `Demand` appears in Models 1 & 2. — GUESS

**Time span & granularity:**
  - `Model_1.csv`: time-related columns: `VA Time`, `Drilling Waiting Time`, `Milling Waiting Time`, `Assembly Waiting Time`
  - `Model_2.csv`: time-related columns: `Part 1 VA Time`, `Drilling Queue Time`, `Part 1 Storage Time`, `Part 2 VA Time`, `Milling Queue Time`, `Part 2 Storage Time`, `Assembly Time`, `Assembly Queue Time`
  - `Model_3.csv`: time-related columns: `Time_Now`, `SKU1_VA_Time`, `SKU1_NVA_Time`, `SKU1_Transport_Time`, `SKU1_Wait_Time`, `SKU1_Other_Time`, `SKU2_VA_Time`, `SKU2_NVA_Time`, `SKU2_Transport_Time`, `SKU2_Wait_Time`, `SKU2_Other_Time`, `SKU3_VA_Time`, `SKU3_NVA_Time`, `SKU3_Transport_Time`, `SKU3_Wait_Time`, `SKU3_Other_Time`, `SKU4_VA_Time`, `SKU4_NVA_Time`, `SKU4_Transport_Time`, `SKU4_Wait_Time`, `SKU4_Other_Time`

## B. File Tree

```
  3000Samplesv3.mat  (33.2 MB)
  Matlab Models\Matlab Models\c1s2ANN.m  (3.7 KB)
  Matlab Models\Matlab Models\c1s2Script.m  (1.2 KB)
  Matlab Models\Matlab Models\c1s4ANN.m  (3.7 KB)
  Matlab Models\Matlab Models\c1s4Script.m  (1.2 KB)
  Matlab Models\Matlab Models\c2s2ANN.m  (3.7 KB)
  Matlab Models\Matlab Models\c2s2Script.m  (1.2 KB)
  Matlab Models\Matlab Models\c2s4ANN.m  (3.7 KB)
  Matlab Models\Matlab Models\c2s4Script.m  (1.2 KB)
  Matlab Models\Matlab Models\c3s2ANN.m  (3.7 KB)
  Matlab Models\Matlab Models\c3s2Script.m  (1.2 KB)
  Matlab Models\Matlab Models\c3s3ANN.m  (3.7 KB)
  Matlab Models\Matlab Models\c3s3Script.m  (1.2 KB)
  Matlab Models\Matlab Models\c4s3ANN.m  (3.7 KB)
  Matlab Models\Matlab Models\c4s3Script.m  (1.2 KB)
  Matlab Models\Matlab Models\c4s4ANN.m  (3.7 KB)
  Matlab Models\Matlab Models\c4s4Script.m  (1.2 KB)
  Matlab Models\Matlab Models\Model1ANN.m  (3.5 KB)
  Matlab Models\Matlab Models\Model1Script.m  (1.2 KB)
  Matlab Models\Matlab Models\Model2ANN.m  (3.5 KB)
  Matlab Models\Matlab Models\Model2ANNAssembly.m  (2.9 KB)
  Matlab Models\Matlab Models\Model2ANNDrilling.m  (3.2 KB)
  Matlab Models\Matlab Models\Model2ANNMilling.m  (3.2 KB)
  Matlab Models\Matlab Models\Model2ScriptAssembly.m  (1.3 KB)
  Matlab Models\Matlab Models\Model2ScriptDrilling.m  (1.3 KB)
  Matlab Models\Matlab Models\Model2ScriptMilling.m  (1.3 KB)
  Matlab Models.zip  (34.0 KB)
  Model 1\Model 1.doe  (412.5 KB)
  Model 1\Model 1.pdf  (99.2 KB)
  Model 1\Model_1.csv  (248.4 KB)
  Model 2\Model 2.doe  (676.5 KB)
  Model 2\Model 2.pdf  (187.1 KB)
  Model 2\Model_2.csv  (360.1 KB)
  Model 3\Model 3.doe  (1.4 MB)
  Model 3\Model 3.pdf  (240.9 KB)
  Model 3\Model_3.csv  (296.7 MB)
  Model 3\ParametersFile.xls  (84.5 KB)
  Readme.txt  (1.8 KB)
```

## C. Tabular File Details

### `Model_1.csv`

- **Size**: 248.4 KB
- **Shape**: (3000, 10)

| Column | dtype | nulls | unique | stats |
| --- | --- | --- | --- | --- |
| `Demand` | int64 | nulls=0 | unique=20 | min=1, max=20, mean=10.44 |
| `Total parts` | int64 | nulls=0 | unique=1364 | min=431, max=1.081e+04, mean=5026 |
| `Parts per hour` | int64 | nulls=0 | unique=387 | min=18, max=450, mean=209.5 |
| `VA Time` | float64 | nulls=0 | unique=2893 | min=8.925, max=9.092, mean=9 |
| `Drilling Waiting Time` | float64 | nulls=0 | unique=2722 | min=0, max=257.6, mean=31.16 |
| `Milling Waiting Time` | float64 | nulls=0 | unique=244 | min=0, max=0.00039, mean=4.572e-05 |
| `Assembly Waiting Time` | float64 | nulls=0 | unique=2874 | min=0, max=244, mean=97.12 |
| `Drilling Util` | float64 | nulls=0 | unique=2838 | min=0.05977, max=1, mean=0.6488 |
| `Milling Util` | float64 | nulls=0 | unique=2977 | min=0.04454, max=0.7521, mean=0.4856 |
| `Assembly Util` | float64 | nulls=0 | unique=2488 | min=0.08893, max=0.9964, mean=0.7759 |

**First 3 rows:**

| `Demand` | `Total parts` | `Parts per hour` | `VA Time` | `Drilling Waiting Time` | `Milling Waiting Time` | `Assembly Waiting Time` | `Drilling Util` | `Milling Util` | `Assembly Util` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7.0 | 3472.0 | 145.0 | 8.998139 | 0.411598 | 0.0 | 2.09364 | 0.482119 | 0.359941 | 0.719475 |
| 10.0 | 1458.0 | 61.0 | 9.000579 | 0.000584 | 0.0 | 0.039279 | 0.202219 | 0.150724 | 0.299455 |
| 2.0 | 1464.0 | 61.0 | 8.984631 | 0.013647 | 0.0 | 0.058197 | 0.204153 | 0.151821 | 0.302495 |

### `Model_2.csv`

- **Size**: 360.1 KB
- **Shape**: (3000, 17)

| Column | dtype | nulls | unique | stats |
| --- | --- | --- | --- | --- |
| `Demand` | int64 | nulls=0 | unique=20 | min=1, max=20, mean=10.5 |
| `Entities In Part 1` | int64 | nulls=0 | unique=1535 | min=659, max=1.546e+04, mean=7537 |
| `Part 1 VA Time` | float64 | nulls=0 | unique=2789 | min=2.958, max=3.028, mean=3 |
| `Drilling Queue Time` | float64 | nulls=0 | unique=2464 | min=0, max=71.02, mean=2.462 |
| `Part 1 Storage Time` | float64 | nulls=0 | unique=3000 | min=0.000789, max=239.9, mean=61.82 |
| `Part 1 Stored` | int64 | nulls=0 | unique=1364 | min=0, max=5925, mean=982.9 |
| `Entities In Part 2` | int64 | nulls=0 | unique=1557 | min=639, max=1.582e+04, mean=7535 |
| `Part 2 VA Time` | float64 | nulls=0 | unique=2772 | min=2.966, max=3.036, mean=3 |
| `Milling Queue Time` | float64 | nulls=0 | unique=2230 | min=0, max=3.339, mean=0.3383 |
| `Part 2 Storage Time` | float64 | nulls=0 | unique=3000 | min=0.00016, max=297.7, mean=62.86 |
| `Part 2 Stored` | int64 | nulls=0 | unique=1379 | min=0, max=6231, mean=980.5 |
| `Entities Out` | int64 | nulls=0 | unique=1781 | min=637, max=9601, mean=6554 |
| `Assembly Time` | float64 | nulls=0 | unique=2795 | min=2.966, max=3.04, mean=3 |
| `Assembly Queue Time` | float64 | nulls=0 | unique=2663 | min=0, max=2.94, mean=1.4 |
| `Drilling Utilization` | float64 | nulls=0 | unique=2996 | min=0.04558, max=0.9999, mean=0.521 |
| `Milling Utilization` | float64 | nulls=0 | unique=2992 | min=0.03305, max=0.8232, mean=0.3919 |
| `Assembly Utilization` | float64 | nulls=0 | unique=2928 | min=0.06716, max=0.9978, mean=0.6835 |

**First 3 rows:**

| `Demand` | `Entities In Part 1` | `Part 1 VA Time` | `Drilling Queue Time` | `Part 1 Storage Time` | `Part 1 Stored` | `Entities In Part 2` | `Part 2 VA Time` | `Milling Queue Time` | `Part 2 Storage Time` | `Part 2 Stored` | `Entities Out` | `Assembly Time` | `Assembly Queue Time` | `Drilling Utilization` | `Milling Utilization` | `Assembly Utilization` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7.0 | 4921.0 | 3.001686 | 0.028374 | 1.52469 | 15.0 | 5138.0 | 3.001738 | 0.00226 | 32.317876 | 232.0 | 4906.0 | 3.007488 | 0.178454 | 0.341792 | 0.267625 | 0.512982 |
| 13.0 | 2160.0 | 2.996812 | 0.0 | 15.424312 | 6.0 | 2265.0 | 2.990346 | 0.0 | 16.641485 | 111.0 | 2154.0 | 3.020345 | 0.0 | 0.149739 | 0.117389 | 0.225907 |
| 3.0 | 2214.0 | 2.99685 | 0.0 | 2.548998 | 14.0 | 2241.0 | 2.985284 | 0.0 | 10.723775 | 41.0 | 2200.0 | 3.002305 | 0.0 | 0.15331 | 0.116007 | 0.230056 |

### `Model_3.csv`

- **Size**: 296.7 MB
- **Shape**: (605620, 78) [stats from first 10000 rows]

**78 columns (grouped by prefix):**

- `Blanking` (6): `Blanking_Util`, `Blanking_SKU1_Queue`, `Blanking_SKU2_Queue`, `Blanking_SKU3_Queue`, `Blanking_SKU4_Queue`, `Blanking_Queue`
- `Cell1` (2): `Cell1_Util`, `Cell1_Queue`
- `Cell2` (2): `Cell2_Util`, `Cell2_Queue`
- `Cell3` (2): `Cell3_Util`, `Cell3_Queue`
- `Cell4` (2): `Cell4_Util`, `Cell4_Queue`
- `Forklift` (4): `Forklift_Util`, `Forklift_Blanking_Queue`, `Forklift_Press_Queue`, `Forklift_Assembly_Queue`
- `Paint1` (2): `Paint1_Util`, `Paint1_Queue`
- `Paint2` (2): `Paint2_Util`, `Paint2_Queue`
- `Press1` (2): `Press1_Util`, `Press1_Queue`
- `Press2` (2): `Press2_Util`, `Press2_Queue`
- `Press3` (2): `Press3_Util`, `Press3_Queue`
- `Press4` (2): `Press4_Util`, `Press4_Queue`
- `Quality` (2): `Quality_Util`, `Quality_Queue`
- `SKU1` (5): `SKU1_VA_Time`, `SKU1_NVA_Time`, `SKU1_Transport_Time`, `SKU1_Wait_Time`, `SKU1_Other_Time`
- `SKU2` (5): `SKU2_VA_Time`, `SKU2_NVA_Time`, `SKU2_Transport_Time`, `SKU2_Wait_Time`, `SKU2_Other_Time`
- `SKU3` (5): `SKU3_VA_Time`, `SKU3_NVA_Time`, `SKU3_Transport_Time`, `SKU3_Wait_Time`, `SKU3_Other_Time`
- `SKU4` (5): `SKU4_VA_Time`, `SKU4_NVA_Time`, `SKU4_Transport_Time`, `SKU4_Wait_Time`, `SKU4_Other_Time`
- `Time` (1): `Time_Now`
- `Warehouse` (3): `Warehouse_2_Queue`, `Warehouse_3_Queue`, `Warehouse_4_Queue`
- `Warehouse1` (1): `Warehouse1_Queue`
- `c` (21): `c_Cycle1`, `c_Cycle2`, `c_Cycle3`, `c_Cycle4`, `c_Cell1_SKU1`, `c_Cell1__SKU2` ...

**15 representative columns (detailed):**

| `Time_Now` | int64 | nulls=0 | unique=1 | min=24, max=24, mean=24 |
| `Blanking_SKU4_Queue` | float64 | nulls=0 | unique=8166 | min=0.02451, max=0.07363, mean=0.04707 |
| `Press1_Queue` | float64 | nulls=0 | unique=10000 | min=29.42, max=158.3, mean=72.28 |
| `Cell2_Util` | float64 | nulls=0 | unique=9621 | min=0.5873, max=0.8221, mean=0.7007 |
| `Cell3_Queue` | int64 | nulls=0 | unique=1 | min=0, max=0, mean=0 |
| `Warehouse_4_Queue` | float64 | nulls=0 | unique=9999 | min=37.04, max=304.7, mean=77.6 |
| `c_Cell1_SKU1` | int64 | nulls=0 | unique=4973 | min=9092, max=2.107e+04, mean=1.47e+04 |
| `c_Cell2__SKU2` | int64 | nulls=0 | unique=1518 | min=1778, max=4109, mean=2872 |
| `c_Cell3__SKU3` | int64 | nulls=0 | unique=2115 | min=2565, max=6074, mean=4262 |
| `c_Cell4__SKU4` | int64 | nulls=0 | unique=4016 | min=2137, max=1.15e+04, mean=6319 |
| `Paint2_Queue` | int64 | nulls=0 | unique=1 | min=0, max=0, mean=0 |
| `Forklift_Assembly_Queue` | float64 | nulls=0 | unique=9993 | min=122.9, max=154.9, mean=138.1 |
| `SKU1_Wait_Time` | float64 | nulls=0 | unique=9844 | min=0.4952, max=2.318, mean=0.725 |
| `SKU2_Wait_Time` | float64 | nulls=0 | unique=9214 | min=0.3729, max=0.5469, mean=0.4234 |
| `SKU3_Wait_Time` | float64 | nulls=0 | unique=9573 | min=0.3807, max=1.025, mean=0.4655 |

**First 3 rows (representative columns):**

| `Time_Now` | `Blanking_SKU4_Queue` | `Press1_Queue` | `Cell2_Util` | `Cell3_Queue` | `Warehouse_4_Queue` | `c_Cell1_SKU1` | `c_Cell2__SKU2` | `c_Cell3__SKU3` | `c_Cell4__SKU4` | `Paint2_Queue` | `Forklift_Assembly_Queue` | `SKU1_Wait_Time` | `SKU2_Wait_Time` | `SKU3_Wait_Time` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24.0 | 0.035849 | 56.540291 | 0.673814 | 0.0 | 75.124986 | 13967.0 | 2851.0 | 4707.0 | 4934.0 | 0.0 | 137.454854 | 0.587305 | 0.441645 | 0.45365 |
| 24.0 | 0.042248 | 74.772823 | 0.660415 | 0.0 | 69.374905 | 15071.0 | 2932.0 | 3852.0 | 7484.0 | 0.0 | 140.757582 | 0.684177 | 0.401993 | 0.473677 |
| 24.0 | 0.040979 | 67.283036 | 0.660288 | 0.0 | 71.720194 | 16603.0 | 2904.0 | 4139.0 | 5575.0 | 0.0 | 142.653371 | 0.898331 | 0.427815 | 0.42409 |

### `ParametersFile.xls`

- **Size**: 84.5 KB
- **Shape**: (50, 5)

| Column | dtype | nulls | unique | stats |
| --- | --- | --- | --- | --- |
| `Module` | str | nulls=49 | unique=1 | top: sch_Driver(1) |
| `Parameter` | str | nulls=44 | unique=6 | top: Velocity(1), Number of units(1), Capacity(1), Duration(1), Forklift Capacity Min(1) |
| `Value` | int64 | nulls=0 | unique=6 | min=0, max=1100, mean=40.9 |
| `Variable` | str | nulls=0 | unique=50 | top: pForklift(1,1)(1), pForklift(2,1)(1), pForklift(3,1)(1), pForklift(4,1)(1), pForklift(5,1)(1) |
| `Default` | float64 | nulls=44 | unique=5 | min=5, max=1100, mean=340.8 |

**First 3 rows:**

| `Module` | `Parameter` | `Value` | `Variable` | `Default` |
| --- | --- | --- | --- | --- |
| nan | Velocity | 5 | pForklift(1,1) | 5.0 |
| nan | Number of units | 8 | pForklift(2,1) | 8.0 |
| sch_Driver | Capacity | 8 | pForklift(3,1) | 8.0 |

### `Readme.txt`

⚠️ Could not read: Error tokenizing data. C error: EOF inside string starting at row 7

## D. Column-Group Role Labels

| Group | Columns (examples) | Likely Role |
| --- | --- | --- |
| Demand / Input params | `Demand` | GUESS: input parameter |
| Utilization | `*_Util`, `*_Utilization` | GUESS: output metric |
| Queue / Waiting | `*_Queue*`, `*Waiting*` | GUESS: output metric |
| Part counters | `c_*`, `Entities*`, `Total parts`, `c_TotalProducts` | GUESS: output metric |
| Time metrics | `VA Time`, `*_VA_Time`, `*_NVA_Time`, `*_Wait_Time`, `Parts per hour` | GUESS: output metric |
| Storage | `*_Storage*`, `*_Stored`, `Warehouse*` | GUESS: output metric |
| Quality | `Quality_Util`, `Quality_Queue` | GUESS: output metric (quality-check station) |
| SKU / Product | `*_SKU*` | GUESS: identifier / categorical |
| Time index | `Time_Now` | GUESS: identifier (simulation clock) |
| Paint / Forklift | `Paint*_Util`, `Forklift_*` | GUESS: output metric |

## E. Non-Tabular / Binary Files

- `3000Samplesv3.mat` (33.2 MB) — `.mat` file. Open with: scipy.io.loadmat or h5py
- `Matlab Models.zip` (34.0 KB) — `.zip` file. Open with: zipfile / 7-Zip
- `Model 1.doe` (412.5 KB) — `.doe` file. Open with: Rockwell Arena Simulation v15 (proprietary)
- `Model 2.doe` (676.5 KB) — `.doe` file. Open with: Rockwell Arena Simulation v15 (proprietary)
- `Model 3.doe` (1.4 MB) — `.doe` file. Open with: Rockwell Arena Simulation v15 (proprietary)

- `Model 1.pdf` (99.2 KB) — PDF process model diagram. Text extraction not attempted (binary).
- `Model 2.pdf` (187.1 KB) — PDF process model diagram. Text extraction not attempted (binary).
- `Model 3.pdf` (240.9 KB) — PDF process model diagram. Text extraction not attempted (binary).
