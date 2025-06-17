# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

---

## Business Understanding

PT Jaya Jaya Maju adalah perusahaan edutech multinasional yang telah beroperasi sejak tahun 2000 dan mempekerjakan lebih dari 1.000 karyawan. Dalam beberapa tahun terakhir, perusahaan menghadapi tantangan serius dalam mempertahankan tenaga kerja dengan **attrition rate mencapai 16.9%**, melebihi ambang wajar industri (~10%). Tingginya tingkat keluar ini tidak hanya berdampak pada meningkatnya biaya rekrutmen, tetapi juga menurunkan stabilitas tim dan mengganggu pengelolaan beban kerja.

Tim HR menyadari pentingnya pendekatan berbasis data untuk memahami *siapa* yang berisiko keluar dan *mengapa*. Tujuan proyek ini adalah untuk membangun sistem analitik yang mampu mengidentifikasi atribut karyawan dengan risiko tinggi, mendukung visualisasi insight strategis, serta membangun model prediksi berbasis machine learning untuk meningkatkan efektivitas keputusan retensi.

---

## Permasalahan Bisnis

- **Tingginya attrition rate** tanpa pemahaman kuantitatif yang mendalam tentang penyebab utamanya.
- **Ketiadaan sistem prediktif** untuk mengenali karyawan yang berpotensi keluar dari perusahaan.
- **Belum adanya dashboard analitik interaktif** untuk memvisualisasikan pola resign dan memantau risiko secara real time.
- **Sulitnya merumuskan strategi retensi** yang akurat karena kurangnya pemetaan risiko pada level individu dan kelompok kerja tertentu.

---

## Cakupan Proyek

- Eksplorasi awal dan pembersihan dataset `employee_data`.
- Seleksi fitur prediktif secara manual berdasarkan insight eksploratif dan domain knowledge.
- Pengembangan model klasifikasi dengan:
  - **Logistic Regression** (baseline interpretable model)
  - **Random Forest**
  - **XGBoost**
  - **Voting Ensemble** (gabungan RF + XGBoost)
- Evaluasi performa model menggunakan metrik: Accuracy, Recall, Precision, dan F1-Score.
- Pengembangan dashboard interaktif menggunakan Metabase.

---

## Persiapan Data

### 📂 Sumber Data

Dataset internal perusahaan dengan nama [`employee_data`](https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/employee/employee_data.csv) diperoleh dari dicoding, terdiri dari **1.470 entri** dan mencakup **35 kolom** informasi terkait demografi, karakteristik pekerjaan, serta riwayat profesional karyawan. Namun, hanya **1.058 observasi** yang memiliki nilai `Attrition` valid (non-null), sehingga proses analisis dan modeling difokuskan pada subset data ini.

Beberapa fitur penting yang digunakan dalam proyek ini antara lain:

- **Atribut Demografis:**  
  - `Age`: Usia karyawan  
  - `Education`: Tingkat pendidikan formal  
  - `MaritalStatus`: Status pernikahan (Single, Married, Divorced)

- **Atribut Pekerjaan:**  
  - `BusinessTravel`: Frekuensi perjalanan bisnis  
  - `DistanceFromHome`: Jarak rumah ke kantor  
  - `JobLevel`: Level jabatan saat ini  
  - `JobRole`: Posisi kerja dalam organisasi  
  - `OverTime`: Status kerja lembur  
  - `MonthlyIncome`: Gaji pokok per bulan  
  - `TotalWorkingYears`: Total pengalaman kerja

- **Target Variabel:**  
  - `Attrition`: Status keluar dari perusahaan (0 = Bertahan, 1 = Keluar)

Kolom identitas unik (`EmployeeId`) serta beberapa kolom konstan seperti `EmployeeCount`, `StandardHours`, dan `Over18` dieksklusi dari proses modeling karena tidak memberikan nilai prediktif.

Dataset ini memberikan kerangka kerja yang cukup kaya untuk menggali insight mendalam terkait faktor-faktor penyebab attrition, baik dari sisi personal maupun struktural organisasi.

### 🛠 Setup Environment
Proyek ini dikembangkan sepenuhnya di Google Colab, yang telah menyediakan lingkungan Python siap pakai di cloud tanpa perlu setup manual lokal.

* Detail Enviromnment:
  - Platform: **Google Colab**
  - Bahasa: **Python 3.11.13**
  - Pustaka yang digunakan:
    - `pandas` - manipulasi data
    - `numpy` - komputasi numerik
    - `scikit-learn` - modeling machine learning
    - `xgboost` - gradient boosting tree
    - `matplotlib`, `seaborn` - visualisasi eksploratif
  - Visualisasi Dashboard: **Metabase** yang terhubung ke Supabase PostgreSQL
  - Penyimpanan data: Database disimpan dan di-query melalui **Supabase PostgreSQL**

Meskipun proyek ini dikembangkan melalui Google Colab, berikut alternatif jika dijalankan di lokal
* Detail Environment (menggunakan anaconda):
  ```bash
  # Membuat environment baru dan mengaktifkannya
  conda create --name hr-attrition python=3.9
  conda activate hr-attrition

  # Instalasi dependencies
  pip install -r requirements.txt

* Menjalankan aplikasi prediksi:
  * Menggunakan file  python
    ```bash
    python attrition_prediction.py

  * Menggunakan streamlit
    ```bash
    streamlit run attrition_prediction.py

  Aplikasi prediksi dapat juga diakses menggunakan streamlit melalui browser: `http://localhost:8501` atau lihat [Demo Online](https://attrition-prediction-hr.streamlit.app/)
---

## Business Dashboard

Dashboard dikembangkan menggunakan **Metabase** dan disusun untuk memudahkan tim HR memahami pola attrition dan memetakan kelompok risiko utama. Dashboard ini bersifat interaktif dan dapat digunakan oleh pengguna non-teknis tanpa perlu menjalankan query manual.

### 🎨 Komponen Visualisasi Utama

1. **Distribusi Attrition Global**  
   Visualisasi pie chart menampilkan proporsi antara karyawan yang bertahan (83.1%) dan yang keluar (16.9%). Ini menjadi indikator awal urgensi manajemen dalam mengatasi turnover.

2. **OverTime vs Attrition**  
   Bar chart menunjukkan bahwa karyawan yang lembur (`OverTime = Yes`) memiliki attrition rate **31.9%**, jauh lebih tinggi dibanding yang tidak lembur (10.8%). Visual ini menegaskan perlunya audit beban kerja.

3. **Attrition Berdasarkan Business Travel**  
   Dari perbandingan tiga kategori (`Travel_Frequently`, `Travel_Rarely`, `Non-Travel`), ditemukan bahwa attrition tertinggi berasal dari `Travel_Frequently` (24.88%). Insight ini mengarah pada evaluasi kebijakan mobilitas kerja.

4. **Distance From Home vs Attrition**  
   Histogram memperlihatkan tren meningkatnya risiko keluar pada karyawan dengan jarak tempat tinggal >15 km. Hal ini mendukung ide penerapan kerja hybrid atau subsidi transportasi.

5. **Monthly Income vs Attrition**  
   Scatter plot income memperlihatkan bahwa attrition cenderung tinggi pada pendapatan <7.000. Namun, attrition juga muncul pada rentang income menengah, menandakan peran persepsi stagnasi karier.

6. **Job Role & Job Level**  
   Visualisasi peran kerja dan level jabatan menunjukkan bahwa posisi seperti *Sales Representative* dan Job Level 1–2 mengalami proporsi attrition paling besar. Hal ini mengarahkan program retensi pada segmen entry-level.

7. **Attrition Berdasarkan Marital Status**  
   Grafik menunjukkan bahwa karyawan yang belum menikah (`Single`) memiliki attrition rate tertinggi (26.7%), diikuti `Divorced` dan `Married`. Ini dapat digunakan sebagai basis segmentasi intervensi retensi berbasis tahapan kehidupan.

Dashboard juga dilengkapi dengan fitur filtering, memungkinkan pengguna melakukan penelusuran spesifik berdasarkan kombinasi variabel seperti `OverTime + DistanceFromHome` atau `JobLevel + BusinessTravel`.

### Cara Akses Dashboard Metabase

   * Buka browser ke [http://localhost:3000](http://localhost:3000)
   * **Login:**
     * Email: `ryorikim06@gmail.com`
     * Password: `190525LaskarAi`
   * Pilih menu **Your personal collection** > **Dewi Rachmawati's Personal Collection** > **edutech-hr-dashboard** untuk melihat visualisasi.
---

## Modeling & Evaluasi

### 🎯 Fitur Terpilih:

Dipilih secara manual dari eksplorasi awal dan pengamatan pola distribusi:
- `OverTime`, `JobRole`, `JobLevel`, `BusinessTravel`, `MonthlyRate`, `DistanceFromHome`, `MaritalStatus`, `NumCompaniesWorked`, `TotalWorkingYears`, `Age`

### 📈 Model yang Dibandingkan:

| Model             | Accuracy | Recall (Attrition) | Precision (Attrition) | F1-Score (Attrition) |
|------------------|----------|---------------------|------------------------|-----------------------|
| LogisticRegression | 83%     | 0.18                | 0.64                   | 0.28                  |
| RandomForest       | 83%     | 0.18                | 0.64                   | 0.28                  |
| XGBoost            | 82%     | 0.31                | 0.52                   | 0.39                  |
| **VotingEnsemble** | **83%** | **0.36**            | **0.54**               | **0.43**              |

### Model Final

**Voting Ensemble (Random Forest + XGBoost)** dipilih sebagai model akhir karena menghasilkan kombinasi terbaik antara akurasi umum dan performa terhadap kelas minoritas (Attrition = 1). Recall yang meningkat berarti model mampu mengenali lebih banyak potensi resign, sedangkan F1-score yang lebih tinggi menunjukkan keseimbangan deteksi dan presisi.

---

## Conclusion

Proyek ini berhasil membangun sistem analitik end-to-end untuk memahami dan memantau attrition karyawan. Hasil eksplorasi dan modeling menunjukkan bahwa atribut seperti `OverTime`, `JobLevel`, `BusinessTravel`, dan `DistanceFromHome` memiliki pengaruh paling signifikan terhadap risiko resign.

Dengan memadukan dashboard interaktif dan model Voting Ensemble, perusahaan kini memiliki alat prediktif dan visual untuk memantau risiko secara real time. Sistem ini dapat digunakan sebagai dasar dalam pengembangan kebijakan kerja yang lebih personal, adil, dan berbasis bukti.

---

## Rekomendasi Action Items

1. **Sistem Deteksi Risiko Resign (Batch Scoring Bulanan)**  
   Jalankan model prediksi secara rutin untuk menghasilkan skor risiko dan jadikan hasilnya dasar prioritas tindakan HR.

2. **Audit Lembur dan Beban Kerja**  
   Karyawan lembur secara rutin menunjukkan risiko keluar paling tinggi — perlu pengaturan ulang workload dan alternatif kompensasi.

3. **Kebijakan Fleksibilitas untuk Karyawan Mobile**  
   Kelompok `Travel_Frequently` dan yang tinggal >15 km dari kantor perlu diberikan opsi hybrid atau dukungan mobilitas.

4. **Program Karier Entry-Level**  
   Job Level 1 dan 2 membutuhkan strategi onboarding, pelatihan, dan jalur promosi yang jelas untuk meningkatkan retensi.

5. **Transparansi Kompensasi dan Promosi**  
   Attrition tidak hanya dipicu oleh nominal gaji, tetapi juga persepsi stagnasi. Audit struktur gaji dan evaluasi sistem promosi dapat meningkatkan kepuasan kerja.
