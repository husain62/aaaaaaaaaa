# Laporan Proyek Machine Learning - Muhammad Husain
## Domain Proyek : Ekonomi

Logam mulia emas memiliki ketahanan yang tinggi untuk disimpan dalam waktu yang lama sehingga tidak mudah rusak walau beberapa tahun berlalu, lunak, tahan korosi, dan mudah ditempa. Emas memiliki supply yang terbatas dan tidak mudah didapat, sementara permintaan terhadap emas tidak pernah berkurang. Hal ini menjadikan nilai atau harga emas cenderung stabil dan naik, sangat jarang turun sehingga dapat digunakan untuk investasi.[_Afida.,2019_](https://download.garuda.kemdikbud.go.id/article.php?article=1428710&val=4706&title=PREDIKSI%20HARGA%20EMAS%20MENGGUNAKAN%20FEED%20FORWARD%20NEURAL%20NETWORK%20DENGAN%20METODE%20EXTREME%20LEARNING%20MACHINE).
Ketika berinvestasi di bursa berjangka berupa emas, yang harus diperhatikan secara hati-hati adalah pergerakan harga emas di pasar fisik. Tinggi rendahnya harga emas dipengaruhi oleh banyak faktor seperti kondisi perekonomian, laju inflasi, penawaran dan permintaan serta masih banyak lagi. Dimungkinkan adanya perubahan faktor-faktor di atas menyebabkan harga dapat naik atau turun. Karena itu perlu prediksi harga emas sehingga bermanfaat bagi investor untuk dapat melihat bagaimana prospek investasi di masa datang.Prediksi harga merupakan salah satu masalah penting, memprediksi harga bisa bervariasi tergantung pada waktu dan informasi dari masa lalu.[_Nugroho.,2015_](https://media.neliti.com/media/publications/137259-ID-penerapan-algoritma-support-vector-machi.pdf).
Melalui analisa yang dilakukan diharapkan mendapat prediksi harga emas yang akurat, yang tidak hanya bermanfaat bagi investor individu tetapi juga penting untuk institusi keuangan, perencana kebijakan, dan perusahaan yang bergantung pada emas sebagai bahan baku. Ketidakakuratan dalam prediksi dapat berujung pada kerugian finansial besar, sementara prediksi yang lebih baik dapat membantu mitigasi risiko, optimalisasi portofolio, dan pengambilan keputusan yang lebih cerdas.

## Business Understanding
### Problem Statemens
Rumusan masalah dari masalah latar belakang diatas adalah :
Bagaimana mengetahui banyak emisi CO2 yang dihasilkan oleh kendaraan berdasarkan riwayat dari fitur-fitur yang ada?
- Dari berbagai variabel yang ada, variabel mana yang paling berpengaruh terhadap harga emas?
- Bagaimana mengetahui prediksi harga emas berdasarkan riwayat dari variabel-variabel yang ada?

### Goals
tujuan untuk menyelesaikan permasalahan diatas adalah:
- Mengetahui variabel yang paling berkorelasi dengan harga emas yang akan datang.
- Membuat model machine learning yang dapat memprediksi berapa harga emas di masa depan secara akurat berdasarkan variabel-variabel yang ada.

### Solution statements
- Melakukan analisis pada data untuk memahami fitur-fitur yang mempengaruhi harga emas, dengan menerapkan teknik visualisasi data guna mengetahui korelasi antar fitur dan memahami hubungan antara data target (label) dan fitur lainnya.
- Menggunakan berbagai algoritma machine learning untuk membandingkan performa model, dengan tujuan mendapatkan model atau algoritma yang memiliki akurasi prediksi tertinggi dalam memperkirakan harga emas.

## Data Understanding

Di website kaggle diketahui bahwa Dataset ini merupakan data historis harga penjualan emas dari tahun ke tahun bervariasi dengan fitur-fitur yang berbeda. Ini adalah versi yang telah dikompilasi dan berisi data selama 10 tahun. Terdapat total 2290 baris dan 6 kolom yang dipisahkan oleh koma. Ada beberapa singkatan yang digunakan untuk menggambarkan fitur-fitur. Saya mencantumkannya di sini, dan hal yang sama dapat ditemukan di lembar Deskripsi Data.
### Informasi Dataset

| Jenis | Keterangan |
| ------ | ------ |
| Title |Gold Price Data |
| Source |[Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/gold-price-data)  |
| Owner |[Debdatta Chatterjee ](https://www.kaggle.com/datasets/altruistdelhite04/gold-price-data) |
| License |Unknown |
| Visibility | 	Publik |
| Usability | 3.53 |

### Variabel-variabel pada  Gold Price Data dataset adalah sebagai berikut:
- Date : Tanggal yang catatan data harga emas 
- SPX : S&P 500 Index
- GLD : exchange-traded fund (ETF) yang dirancang untuk melacak harga emas fisik.
- USO : ETF yang melacak harga minyak mentah.
- SLV : ETF yang melacak harga perak.
- EUR/USD : Nilai tukar mencerminkan kekuatan relatif antara euro dan dolar AS.

### Exploratory Data Analysis - Deskripsi Variabel
| #   | Column   | Non-Null Count | Dtype   |
|-----|----------|----------------|---------|
| 0   | Date     | 2290 non-null  | object  |
| 1   | SPX      | 2290 non-null  | float64 |
| 2   | GLD      | 2290 non-null  | float64 |
| 3   | USO      | 2290 non-null  | float64 |
| 4   | SLV      | 2290 non-null  | float64 |
| 5   | EUR/USD  | 2290 non-null  | float64 |

Dari tabel diatas didapat informasi banyak fitur categorical dan numerik :
- Fitur Categorical (Object):
Date: Tipe data object (kemungkinan berupa tanggal, perlu diubah ke tipe datetime untuk analisis lebih lanjut).
- Fitur Numeric (Float64):
SPX: Indeks S&P 500.
GLD: Harga emas (Gold).
USO: Harga minyak mentah (Crude Oil).
SLV: Harga perak (Silver).
EUR/USD: Nilai tukar Euro terhadap Dolar AS.


| Column   | Count | Mean         | Std Dev     | Min         | 25%         | 50%         | 75%         | Max         |
|----------|-------|--------------|-------------|-------------|-------------|-------------|-------------|-------------|
| SPX      | 2290  | 1654.315776  | 519.111540  | 676.530029  | 1239.874969 | 1551.434998 | 2073.010070 | 2872.870117 |
| GLD      | 2290  | 122.732875   | 23.283346   | 70.000000   | 109.725000  | 120.580002  | 132.840004  | 184.589996  |
| USO      | 2290  | 31.842221    | 19.523517   | 7.960000    | 14.380000   | 33.869999   | 37.827501   | 117.480003  |
| SLV      | 2290  | 20.084997    | 7.092566    | 8.850000    | 15.570000   | 17.268500   | 22.882500   | 47.259998   |
| EUR/USD  | 2290  | 1.283653     | 0.131547    | 1.039047    | 1.171313    | 1.303297    | 1.369971    | 1.598798    |

Table diatas memberikan informasi statistik pada masing-masing kolom, antara lain:
- Count adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom.
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval - dalam empat bagian sebaran yang sama.
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.

| **Jumlah Baris** | **Jumlah Kolom** |
|-------------------|------------------|
| 2290             | 5            |

### Exploratory Data Analysis - Missing Value 
**Mengecek missing value, Duplikat.**
| **Data Duplikat** |
|-------------------|
| 0        |
Tidak terdapat data duplikat

| **Kolom**  | **Jumlah Data Hilang (Null)** |
|------------|-------------------------------|
| SPX        | 0                             |
| GLD        | 0                             |
| USO        | 0                             |
| SLV        | 0                             |
| EUR/USD    | 0                             |
Tidak terdapat missing value

### Exploratory Data Analysis - Univariate Analysis

**Univariate Analysis - Fitur Numerik**
![Screenshot 2024-11-26 183743](https://github.com/user-attachments/assets/5d10cf61-89ae-4907-8ff0-5bb959e137a6)

Pada histogram diatas dapat disimpulkan bahwa:

- Variabel GLD dan SPX menunjukkan distribusi data yang mendekati normal.
- USO memiliki distribusi yang tidak simetris dan mencerminkan nilai yang       skewed, mungkin akibat faktor tertentu dalam pasar minyak.
- Variabel EUR/USD dan price_trend cenderung memiliki beberapa cluster atau puncak dalam distribusinya

![Screenshot 2024-11-26 190546](https://github.com/user-attachments/assets/ddf0ef03-3a3f-43db-a71a-d940b7e23c19)
Berdasarkan matriks korelasi untuk fitur numerik yang ditampilkan:
- GLD (Harga Emas) memiliki hubungan yang sangat erat dengan SLV (Harga Perak), menjadikan SLV salah satu prediktor penting untuk harga emas.
- USO (Harga Minyak) dan EUR/USD memiliki hubungan yang kuat, menunjukkan bahwa pergerakan harga minyak sangat relevan dengan nilai tukar mata uang.
- SPX (S&P 500 Index) lebih independen terhadap GLD tetapi memiliki hubungan negatif yang signifikan dengan EUR/USD dan USO.

## Data Preparation
**Train-Test-Split**
Sebelum melakukan train-test-split, langkah awalnya adalah memisahkan antara fitur dan label. Variabel x digunakan untuk menyimpan fitur yang terdiri dari:
- SPX
- USO
- SLV
- EUR/USD

Sedangkan variabel y digunakan untuk menyimpan label yaitu GLD.

Selanjutnya, dilakukan train-test-split dengan pembagian data sebesar 80:20 antara data latih (train) dan data uji (test).

**Jadi mengapa perlu dilakukan data prepration?**
- Memisahkan data menjadi set pelatihan dan pengujian memungkinkan kita untuk mengevaluasi kinerja model pada data yang tidak pernah dilihat sebelumnya. Ini memberikan gambaran yang lebih akurat tentang seberapa baik model dapat menggeneralisasi ke data baru.

## Modeling
Model machine learning yang digunakan untuk masalah ini terdiri dari 2 model yaitu:
**1 DecisionTreeRegressor**

Cara Membuat Model:

from sklearn.tree import DecisionTreeRegressor

Mengimpor class DecisionTreeRegressor dari pustaka scikit-learn.
dt_regressor = DecisionTreeRegressor(min_samples_leaf=1, min_samples_split=2, max_depth=None, max_features=None, random_state=None)

Membuat objek dari class DecisionTreeRegressor dengan parameter-parameter yang ditentukan.

Penjelasan Parameter:

min_samples_leaf=1: Menentukan jumlah minimum sampel yang harus ada di simpul daun. Jika diatur ke 1 (default), setiap simpul daun bisa memiliki minimal 1 sampel.

min_samples_split=2: Menentukan jumlah minimum sampel yang diperlukan untuk membagi simpul. Jika ada kurang dari 2 sampel di sebuah simpul, maka simpul tersebut tidak akan dibagi lebih lanjut.

max_depth=None: Menentukan kedalaman maksimum pohon. Jika diatur ke None, pohon akan tumbuh sampai semua simpul daun murni, atau sampai jumlah sampel di simpul kurang dari min_samples_split.

max_features=None: Menentukan jumlah maksimum fitur yang dipertimbangkan pada setiap pembagian (split). Jika diatur ke None, semua fitur akan dipertimbangkan.

random_state=None: Menentukan seed untuk generator bilangan acak agar hasil dapat diulang. Jika None, hasil tidak dijamin dapat diulang karena angka acak digunakan tanpa seed tetap.

dt_regressor.fit(X_train, y_train)

Memanggil metode fit untuk melatih model DecisionTreeRegressor menggunakan data pelatihan X_train (fitur) dan y_train (target).

X_train: Matriks atau array berisi data fitur yang digunakan untuk melatih model. Setiap baris mewakili satu sampel, dan setiap kolom mewakili satu fitur.
y_train: Label atau nilai target yang sesuai dengan data fitur dalam X_train. Ini adalah nilai yang ingin diprediksi oleh model.
Selama proses pelatihan, model akan membangun pohon keputusan dengan membagi data ke dalam beberapa simpul berdasarkan pembagian optimal dari fitur-fitur, yang meminimalkan kesalahan prediksi.

Cara Kerja Model:
Decision Tree Regressor bekerja dengan membuat pohon keputusan berdasarkan fitur-fitur dataset. Setiap simpul pada pohon dipisahkan menjadi beberapa cabang dengan mencari pemisahan (split) yang terbaik, yang diukur berdasarkan pengurangan varian atau peningkatan informasi. Proses ini berlanjut sampai tidak ada lagi pemisahan yang dapat dilakukan, atau sampai semua simpul daun memiliki kurang dari jumlah minimum sampel untuk dibagi.

**2 RandomForestRegressor**
Cara Membuat Model:
from sklearn.ensemble import RandomForestRegressor
Mengimpor class RandomForestRegressor dari pustaka scikit-learn.
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
Membuat objek dari class RandomForestRegressor dengan parameter yang ditentukan.
Penjelasan Parameter:

n_estimators=50: Menentukan jumlah pohon keputusan yang digunakan dalam hutan (forest).

max_depth=16: Menentukan kedalaman maksimum pohon-pohon keputusan dalam hutan untuk menghindari overfitting.
random_state=55: Seed generator untuk memastikan hasil dapat diulang.
n_jobs=-1: Menentukan jumlah core yang digunakan. Jika diatur ke -1, model akan menggunakan semua core yang tersedia.
RF.fit(X_train, y_train)
Memanggil metode fit untuk melatih model RandomForestRegressor menggunakan data pelatihan X_train dan y_train.
X_train: Matriks fitur yang digunakan untuk pelatihan.
y_train: Nilai target yang sesuai dengan X_train.

Cara Kerja Model:
Random Forest bekerja dengan membangun banyak pohon keputusan (decision trees) secara acak dan menggabungkan hasilnya untuk membuat prediksi. Setiap pohon dilatih pada subset acak dari data pelatihan dan subset acak dari fitur, sehingga mengurangi overfitting. Hasil dari semua pohon kemudian diambil rata-ratanya (regresi) atau suara mayoritas (klasifikasi) untuk menghasilkan prediksi akhir.

**Kelebihan dan Kekurangan model tersebut.**
- Decision Tree Regressor

Kelebihan:
1.Mudah dipahami dan diinterpretasikan.
2.Tidak memerlukan normalisasi data.
3.Dapat menangani data dengan hubungan non-linear dan interaksi antar fitur.
Menyediakan representasi grafis yang jelas dan mudah dimengerti.

Kekurangan:
1.Rentan terhadap overfitting, terutama jika pohon terlalu dalam.
2.Sensitif terhadap perubahan kecil dalam data.
3.Tidak dapat menggeneralisasi dengan baik pada data yang tidak terlihat.

- Random Forest Regressor

Kelebihan:
1.Menggabungkan hasil dari banyak pohon keputusan untuk menghasilkan prediksi yang lebih akurat dan robust.
2.Memiliki mekanisme built-in untuk menangani overfitting.
3.Dapat menangani data yang hilang dengan baik.

Kekurangan:
1.Lebih lambat dalam pelatihan dan prediksi dibandingkan model sederhana seperti linear regression.
2.Model yang dihasilkan sulit untuk diinterpretasikan secara langsung.
3.Memerlukan lebih banyak memori dibandingkan model yang lebih sederhana.

**Mengapa menggunakan 2 model tersebut?.**
- Decision Tree Regressor Model ini mudah dipahami dan diinterpretasikan. Decision Tree dapat menangani data dengan hubungan non-linear dan interaksi antar fitur tanpa perlu transformasi fitur. Selain itu, ia tidak memerlukan normalisasi data.
- Random Forest Regressor menggabungkan hasil dari banyak pohon keputusan untuk menghasilkan prediksi yang lebih akurat dan robust. Ia juga memiliki mekanisme built-in untuk menangani overfitting dan dapat menangani data yang hilang dengan baik.

## Evaluation
Metrik evaluasi yang digunakan dalam analisis ini adalah R-squared, juga dikenal sebagai koefisien determinasi, adalah ukuran yang menunjukkan seberapa baik model machine learning menjelaskan variabilitas data target (nilai yang diprediksi). Nilai R² berkisar antara 0 hingga 1, dan semakin mendekati 1, semakin baik model dalam menjelaskan variasi data.
![Screenshot 2024-11-26 195033](https://github.com/user-attachments/assets/ffe7a55d-87d7-4ed2-b7de-1c6a383c0a77)

yi = Nilai observasi (data asli).
yi^ = Nilai prediksi oleh model.
y- = Rata-rata nilai asli.
n = Jumlah data.

**Evaluasi Model Machine Learning yang diusulkan**
- Berikut adalah hasil yang diperoleh dari metrik ini, diurutkan dari kesalahan terkecil hingga terbesar:

| **Model** | **R-Squared** |
|-------------------|------------------|
|DecisionTreeRegressor           | 0.9924842369293052        |
|Random Forest      |  0.9970881167928056       |

Dari tabel di atas, dapat dilihat bahwa setiap model menghasilkan prediksi yang bervariasi untuk setiap nilai aktual (y_true). Random Forest menunjukkan tampaknya menghasilkan prediksi yang lebih konsisten dengan nilai 0.988663782763203, dengan kesalahan relatif yang lebih kecil pada nilai prediksi dibandingkan dengan model lainnya. 

![Screenshot 2024-11-26 200340](https://github.com/user-attachments/assets/16f9b659-ac40-465f-981a-327d519381f2)
Plot diatas adalah hasil perbandingan antara harga asli dengan haraga prediksi menggunakan model Random Forrest

## Kesimpulan

Dari hasil analisis dan evaluasi yang telah dilakukan, dapat disimpulkan bahwa model yang diusulkan berhasil menjawab kedua rumusan masalah yang diajukan. Pertama, melalui analisis unvariate, model mengidentifikasi bahwa Harga Emas Hari Sebelumnya, Nilai Tukar Mata Uang (USD), Indeks Harga Saham, dan Harga Minyak Mentah merupakan fitur-fitur yang paling berpengaruh terhadap prediksi harga emas. Kedua, model mampu memprediksi harga emas berdasarkan fitur-fitur tersebut dengan akurasi yang baik.
Dalam pencapaian goals, model yang dikembangkan mencapai tujuan yang diharapkan, yaitu  membuat model machine learning yang dapat memprediksi harga emas secara akurat. Hasil evaluasi menunjukkan bahwa model Random Forest  memiliki kesalahan R squared yang rendah, menandakan bahwa efektif dalam melakukan prediksi.
Dampak dari solusi yang diterapkan, termasuk analisis visualisasi data dan penerapan berbagai algoritma machine learning, memberikan pemahaman yang lebih mendalam tentang hubungan antar fitur dan harga emas. Ini tidak hanya membantu dalam memahami struktur data, tetapi juga dalam pengambilan keputusan strategis. Misalnya, investor atau pelaku pasar dapat menggunakan informasi ini untuk membuat keputusan investasi yang lebih tepat dengan mempertimbangkan faktor-faktor yang memengaruhi harga emas.









