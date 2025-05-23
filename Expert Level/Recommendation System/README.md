# Laporan Proyek Machine Learning - Nugroho Adi Wirapratama

## Project Overview

Di era digital saat ini, informasi mengenai tempat wisata dapat dengan mudah diakses secara online. Namun, banyaknya pilihan sering kali membuat pengguna kebingungan dalam menentukan destinasi yang sesuai dengan preferensi mereka. Untuk itu, sistem rekomendasi destinasi wisata menjadi sangat relevan dan dibutuhkan sebagai solusi dalam membantu pengguna menemukan tempat yang paling sesuai dengan minat dan kebutuhan mereka.

Proyek ini bertujuan untuk membangun sistem rekomendasi wisata berbasis Content-Based Filtering (CBF) dan Collaborative Filtering (CF) dengan pendekatan machine learning, yang mampu memberikan rekomendasi secara personal kepada pengguna berdasarkan deskripsi tempat, kategori, lokasi, maupun interaksi pengguna sebelumnya seperti rating.

**Mengapa proyek ini harus diselesaikan ?**

Pentingnya proyek ini diselesaikan tidak hanya terletak pada nilai praktisnya dalam meningkatkan pengalaman pengguna saat merencanakan perjalanan, tetapi juga dalam mengurangi waktu pencarian dan mendukung promosi destinasi lokal yang relevan. Dengan adanya sistem rekomendasi, pengguna akan lebih mudah menemukan tempat-tempat yang mungkin belum populer, namun sesuai dengan ketertarikan mereka.
  
**Referensi:**
- Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer. [Link](https://www.researchgate.net/publication/227268858_Recommender_Systems_Handbook)

**Dataset :** https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination

---

## Business Understanding
Pengguna sering kali dihadapkan pada pilihan destinasi wisata yang sangat beragam, namun tidak selalu mudah menemukan tempat yang benar-benar sesuai dengan preferensi pribadi mereka. Sistem rekomendasi hadir sebagai solusi untuk meningkatkan pengalaman pengguna, mempercepat proses pengambilan keputusan, dan meningkatkan keterlibatan mereka terhadap platform digital pariwisata.

Memahami secara mendalam permasalahan bisnis serta tujuan yang ingin dicapai menjadi langkah awal yang penting dalam pengembangan proyek ini.

### Problem Statements
- Bagaimana cara merekomendasikan destinasi wisata yang relevan bagi pengguna berdasarkan karakteristik dari tempat-tempat wisata itu sendiri?
- Bagaimana memanfaatkan riwayat interaksi dan penilaian pengguna lain untuk memberikan rekomendasi wisata yang personal?

### Goals
- Mengembangkan sistem rekomendasi berbasis konten (Content-Based Filtering) yang dapat menyarankan destinasi serupa berdasarkan atribut seperti kategori, kota, dan deskripsi tempat wisata.
- Mengimplementasikan sistem rekomendasi berbasis kolaborasi (Collaborative Filtering) yang memanfaatkan data rating pengguna untuk memberikan rekomendasi yang sesuai dengan preferensi pengguna lain yang serupa.

### Solution Approach
Untuk menyelesaikan masalah dan mencapai tujuan di atas, proyek ini akan menggunakan dua pendekatan sistem rekomendasi:

1. **Content-Based Filtering (CBF)**
Menggunakan informasi atau atribut dari masing-masing tempat wisata, sistem akan menghitung kesamaan antar destinasi berdasarkan deskripsi, kategori, dan kota. Rekomendasi akan diberikan berdasarkan kemiripan dengan tempat yang sebelumnya disukai pengguna.
    - **Teknik**: TF-IDF untuk deskripsi, cosine similarity untuk menghitung kemiripan
    - **Fokus**: Karakteristik konten dari tempat wisata

2. **Collaborative Filtering (CF)**
Sistem ini menggunakan rating pengguna lain yang memiliki preferensi serupa untuk merekomendasikan tempat wisata. pendekatan ini menggunakan teknik Matrix Factorization (misalnya SVD atau ALS) untuk memetakan pengguna dan destinasi ke dalam ruang laten. Dengan demikian, sistem dapat memprediksi rating yang mungkin diberikan oleh pengguna terhadap destinasi yang belum mereka kunjungi, lalu memberikan rekomendasi berdasarkan prediksi tersebut.
    - **Teknik**: Matrix Factorization (contoh: Singular Value Decomposition)
    - **Fokus**: Pola laten dari interaksi antara pengguna dan destinasi

---

## Data Understanding
Dataset didapatkan dari platform kaggle. Link dataset : **https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination**

Terdapat 4 dataset .csv yang tersedia yaitu
- tourism_ with _id.csv : berisi informasi tentang objek wisata di 5 kota besar di Indonesia dengan total sekitar 400 tempat wisata.
- user.csv : berisi data pengguna dummy untuk membuat fitur rekomendasi berdasarkan pengguna.
- tourism_rating.csv : berisi 3 kolom, yaitu pengguna, tempat wisata, dan rating yang diberikan, digunakan untuk membuat sistem rekomendasi berdasarkan rating.
- package_tourism.csv : berisi rekomendasi tempat wisata terdekat berdasarkan waktu, biaya, dan rating.

File .csv yang digunakan untuk pembuatan sistem rekomendasi ini hanya **tourism_ with _id.csv** dan **tourism_rating.csv**. Kedua file tersebut dilakukan merge dataset.

### 1. Variabel pada dataset 
#### tourism_ with _id.csv
- Place_Id : id dari Place_Name.
- Place_Name : nama tempat destinasi.
- Description : teks deskripsi tentang destinasi.
- Category : kategori dari destinasi.
- City : lokasi kota dari destinasi.
- Price : harga tiket masuk destinasi.
- Rating : rata-rata rating yang diberikan destinasi.
- Time_Minutes : rata-rata waktu pengunjung
- Coordinate : koordinat longitude & latitude destinasi.
- Lat : koordinat latitude destinasi wisata.
- Long : koordinat longitude destinasi wisata.
- Unnamed: 11 : unknown information
- Unnamed: 12 : unknown information

| Place_Id | Place_Name | Description | Category | City | Price | Rating | Time_Minutes	| Coordinate | Lat | Long | Unnamed: 11 | Unnamed: 12|
| ------ |------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------|
| 179 | Candi Ratu Boko | Situs Ratu Baka atau Candi Boko (Hanacaraka:ꦕꦤ... | Budaya | Yogyakarta | 75000 | 4.6 | 15.0 | {'lat': -6.1753924, 'lng': 106.8271528} | -6.175392 | 106.827153 | NaN | 1 |
| 344 | Pantai Marina | Pantai Marina (bahasa Jawa: ꦥꦱꦶꦱꦶꦂ​ꦩꦫꦶꦤ, trans... | Bahari | Semarang | 3000 | 4.1 | 90.0 | {'lat': -6.137644799999999, 'lng': 106.8171245} | -6.137645 | 106.817125 | NaN | 2 |
| 5   | Atlantis Water Adventure | Atlantis Water Adventure atau dikenal dengan A... | Taman Hiburan | Jakarta | 94000 | 4.5 | 360.0 | {'lat': -6.125312399999999, 'lng': 106.8335377} | -6.125312 | 106.833538 | NaN | 3 |
| 373 | Museum Kereta Ambarawa | Museum Kereta Api Ambarawa (bahasa Inggris: In... | Budaya | Semarang | 10000 | 4.5 | NaN | {'lat': -6.302445899999999, 'lng': 106.8951559} | -6.302446 | 106.895156 | NaN | 4 |
| 101 | Kampung Wisata Sosro Menduran | Kampung wisata Sosromenduran merupakan kampung... | Budaya | Yogyakarta | 0 | 4.0 | 60.0 | {'lat': -6.12419, 'lng': 106.839134} | -6.124190 | 106.839134 | NaN | 5 |

#### tourism_rating.csv
- User_Id : user_id.
- Place_Id : id dari Place_Name.
- Place_Ratings : rating yang diberikan User_Id terhadap destinasi.

| User_Id | Place_Id | Place_Ratings |
| ------ | ------ | ------ |
| 1 | 179 |	3 |
| 1 | 344 |	2 |
| 1 | 5 | 5 |
| 1 | 373 |	3 |
| 1 | 101 |	4 |

### 2. Dataset information
#### tourism_ with _id.csv
![image](https://github.com/user-attachments/assets/8eda1bd0-56cb-44d2-a486-4bed00365a0f)

Dataset **tourism_ with _id.csv** memiliki **437 baris** dan **7 kolom**. Untuk tipe data pada dataset sudah sesuai.

#### tourism_rating.csv
![image](https://github.com/user-attachments/assets/979a6102-019a-4652-ae86-5980bcb96d44)

Dataset **tourism_rating.csv** memiliki **10000 baris** dan **3 kolom**. Untuk tipe data pada dataset sudah sesuai.

### 3. NaN Value
 ![image](https://github.com/user-attachments/assets/7a040c01-bf5c-46ac-b5cc-2efbba5cc9c4) | ![image](https://github.com/user-attachments/assets/871282d7-dece-4412-946f-da4456699170)
|:--:|:--:|

- Gambar kanan merupakan data **tourism_ with _id.csv** -> Terdapat NaN value pada kolom **Time_Minutes** dan **Unnamed: 11**
- Gambar kiri merupakan data **tourism_rating.csv** -> Tidak ada data null.

### 4. Data Duplikat

![image](https://github.com/user-attachments/assets/ff494c9d-5fb2-4d37-8ef7-9334e04fc341) | ![image](https://github.com/user-attachments/assets/ee610911-dfa7-4b1f-a523-40e681cd604a)
|:--:|:--:|

- Pada data **tourism_with_id.csv** tidak terdapat data duplikat.
- Pada data **tourism_rating.csv** terdapat **79 data duplikat** yang benar-benar sama.

### 5. Top 10 Places by Average Rating
![image](https://github.com/user-attachments/assets/6c97f0fd-c30d-4459-b11d-211523187754)

Freedom Library menduduki peringkat pertama dengan rating tertinggi, diikuti oleh Desa Wisata Sungai Code Jogja Kota dan Kauman Pakualaman Yogyakarta.

### 6. Scatter Plot: Price vs Rating
![image](https://github.com/user-attachments/assets/776f72d7-af5b-4844-9018-b485ce74d82e)

Dari scatter plot, terlihat bahwa meskipun harga bervariasi, sebagian besar destinasi memiliki rating tinggi (di atas 4), namun tidak ada pola yang jelas antara harga dan rating.

![image](https://github.com/user-attachments/assets/d5eeeece-2433-486f-a330-7f70aaea2e8f)

Dalam analisis regresi, hasil menunjukkan bahwa koefisien untuk harga sangat kecil (7.296e-08), dan p-value sebesar 0.628 menunjukkan bahwa harga tidak berpengaruh signifikan terhadap rating destinasi wisata. Nilai R-squared yang sangat rendah (0.001) juga menegaskan bahwa harga hanya menjelaskan sedikit sekali variasi dalam rating.

**Harga tidak menjadi faktor yang signifikan dalam menentukan rating destinasi wisata.**

### 7. Barplot Distribusi Rating
![image](https://github.com/user-attachments/assets/7a907eb2-7dfc-4723-9ef6-f69701a45e79)

Dari grafik ini, dapat dilihat bahwa distribusi pemberian rating oleh user relatif merata, di mana setiap rating (1 hingga 5) memiliki jumlah yang hampir sama, sekitar 2000, dengan sedikit perbedaan antara setiap kategori rating.

---

## Data Preparation
Ada beberapa tahapan untuk data preparation sebelum memulai untuk membuat sistem rekomendasi.

### 1. Merge dataset
Melakukan penggabungan dataset antara **tourism_with_id.csv** dan **tourism_rating.csv**

### 2. Menghapus kolom yang tidak relevan
- Menghapus kolom yang tidak relevan untuk pembuatan model sistem rekomendasi, yaitu:
  - ['Unnamed: 11','Unnamed: 12', 'Time_Minutes', 'Coordinate',	'Lat',	'Long', 'Rating', 'Price']
- Pada Rating menggunakan kolom Place_Ratings dari dataset tourism_rating.csv karena menggambarkan interaksi user dalam memberikan rating terhadap destinasi, hal ini untuk keperluan model **Collaborative Filtering**.

### 2. NaN value
Sebelumnya ada NaN value untuk data **tourism_with_id.csv** tetapi dikarenakan NaN berada di kolom **Time_Minutes** dan **Unnamed: 11** sedangkan 2 kolom tersebut tidak digunakan, maka handling NaN value tidak diperlukan.

### 3. Drop/delete duplikat value
Terdapat 79 baris duplikat, karena baris-baris ini sepenuhnya identik, maka dihapus untuk menghindari bias pada hasil pemodelan.

### 3. Processing text

Processing text dilakukan pada kolom **Description**, **City**, dan **Category**. Tahapan processing text yang dilakukan adalah sebagai berikut:
- Lowercasing
- Penghapusan tanda baca dan angka
- Penghapusan stopwords Bahasa Indonesia
- Stemming menggunakan Sastrawi

Tujuan dari preprocessing ini adalah untuk menyederhanakan teks dan meningkatkan kualitas perhitungan kemiripan antar destinasi wisata.

### 4. Pemisahan data untuk Content Based Filtering dan Collaborative Filtering

Pembagian data ini dikarenakan bentuk dan jenis data yang digunakan untuk kedua metode tersebut berbeda, oleh karena itu dilakukan proses yang berbeda untuk masing-masing pendekatan: 

- **Content Based Filtering**

Untuk pendekatan Content Based Filtering, digunakan atribut deskriptif seperti **Description**, **Category**, dan **City**. Duplikasi berdasarkan **Place_Id** dihapus agar tidak terjadi redundansi destinasi.

Kemudian, TF-IDF digunakan untuk mengubah deskripsi tempat menjadi representasi numerik, dan cosine similarity digunakan untuk menghitung kemiripan antar destinasi.
```python
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_cbf['Description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

- **Collaborative Filtering**

Untuk pendekatan Collaborative Filtering berbasis Matrix Factorization, digunakan atribut User_Id, Place_Name, dan Rating. Informasi deskriptif tempat tidak diperlukan karena fokus utama adalah interaksi pengguna terhadap destinasi.

Dataset untuk **Collaborative Filtering** dilakukan **split dataset train dan test dengan ratio 8:2**.

Alih-alih menghitung kemiripan antar pengguna secara langsung, pendekatan ini membentuk user-item matrix dan menerapkannya pada model Matrix Factorization (misalnya menggunakan algoritma SVD). Model ini akan mempelajari representasi laten dari pengguna dan destinasi untuk memprediksi rating yang hilang dan merekomendasikan tempat yang paling relevan.

Pembentukan user-item matrix
```python
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_cf[['User_Id', 'Place_Name', 'Place_Ratings']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)
```

---

## Modeling
Untuk menyelesaikan permasalahan sistem rekomendasi destinasi wisata, proyek ini menggunakan dua pendekatan utama:
- **Content-Based Filtering (CBF)**
- **Collaborative Filtering (CF)**

Setiap pendekatan dikembangkan untuk memberikan top-N recommendation, yakni daftar destinasi wisata yang paling relevan berdasarkan metode yang digunakan.

### 1. Content-Based Filtering (CBF)
CBF bekerja dengan menganalisis kemiripan antar destinasi berdasarkan kontennya. Dalam proyek ini, atribut yang digunakan meliputi:
- Deskripsi tempat wisata
- Kategori wisata (misalnya: pantai, pegunungan, budaya, dll.)
- Kota tempat wisata berada

Deskripsi diolah menggunakan TF-IDF Vectorizer untuk mengubah teks menjadi vektor numerik, lalu dihitung kemiripannya menggunakan cosine similarity. Skor akhir ditentukan dari kombinasi bobot berikut:
- 60% kemiripan deskripsi
- 25% kesamaan kategori
- 15% kesamaan kota
```
final_score = (0.6 * description_score) + (0.25 * category_score) + (0.15 * city_score)
```
Sistem kemudian menyaring destinasi dengan skor tertinggi sebagai top-N recommendation. Contohnya, jika pengguna tertarik pada “Pantai Kuta”, maka sistem akan merekomendasikan destinasi serupa berdasarkan kontennya seperti "Pantai Seminyak", "Jimbaran Beach", dan sebagainya.

**Contoh Output (Top-10 Rekomendasi):**

Sebagai contoh, jika pengguna tertarik pada destinasi "Pasar Seni", sistem akan mengembalikan rekomendasi seperti:
- Destinasi: Pasar Seni
- Kategori: pusat belanja
- Kota: jakarta
- Deskripsi: pasar seni pusat rajin seni wadah seniman bakat salur rajin seni milik seniman temu kolektor seni usaha bagi informasi pasar produk milik diri leta batu bang ali sadikin gubernur dki jakarta tepat juli resmi gubernur h tjokropanolo tanggal desember resmi pasar seni sambut positif cinta seni

10 Rekomendasi:
| Recommended Destination | Category | Category | Description Similarity | Category Score | City Score | Total Score |
| ------ | ------ |------ | ------ | ------ | ------ | ------ |
| Pasar Tanah Abang | pusat belanja | jakarta | 0.2060 | 1 | 1 | 0.5236 |
| Pasar Taman Puring | pusat belanja | jakarta | 0.1362 | 1 | 1 | 0.4817 |
| Mall Thamrin City | pusat belanja | jakarta | 0.0530 | 1 | 1 | 0.4318 |
| Plaza Indonesia | pusat belanja | jakarta | 0.0525 | 1 | 1 | 0.4315 |
| Pecinan Glodok | pusat belanja | jakarta | 0.0429 | 1 | 1 | 0.4257 |
| Pasar Petak Sembilan | pusat belanja | jakarta | 0.0317 | 1 | 1 | 0.4190 |
| Kawasan Kuliner BSM | pusat belanja | jakarta | 0.0216 | 1 | 1 | 0.4130 |
| Wisata Kuliner Pecenongan | pusat belanja | jakarta | 0.0147 | 1 | 1 | 0.4088 |
| Pasar Baru | pusat belanja | bandung | 0.2566 | 1 | 0 | 0.4040 |
| Grand Indonesia Mall | pusat belanja | jakarta | 0.0065 | 1 | 1 | 0.4039 |

### 2. Collaborative Filtering (CF)
Berbeda dengan CBF, pendekatan CF menggunakan informasi dari interaksi pengguna lain. Sistem mengasumsikan bahwa pengguna dengan preferensi serupa akan menyukai tempat yang sama. Jika pengguna A memiliki preferensi yang mirip dengan pengguna B, maka destinasi yang disukai B tapi belum dijelajahi A dapat direkomendasikan ke A.

Pendekatan yang digunakan adalah Matrix Factorization, tepatnya Singular Value Decomposition (SVD). Teknik ini memetakan pengguna dan tempat wisata ke dalam ruang fitur laten (latent factors), lalu memprediksi rating yang mungkin diberikan oleh pengguna terhadap tempat-tempat yang belum pernah mereka kunjungi.

Langkah-langkah utama:
- Membuat user-item matrix berdasarkan rating yang diberikan pengguna terhadap destinasi wisata.
- Menggunakan algoritma SVD untuk melakukan dekomposisi matriks menjadi representasi vektor pengguna dan item.
- Menghasilkan prediksi skor untuk semua kombinasi user-item yang belum pernah diinteraksikan.
- Memberikan top-N rekomendasi berdasarkan skor prediksi tertinggi.

**Contoh Output (Top-10 Rekomendasi untuk user_id 1):**
| Recommended Place | Estimated Score |
| ------ | ------ |
| Gunung Lalakon | 13.336041 |
| Masjid Agung Trans Studio Bandung | 13.070383 |
| Museum Nike Ardilla | 12.321744 |
| Food Junction Grand Pakuwon | 12.251095 |
| Pantai Parangtritis | 11.721672 |
| Kebun Tanaman Obat Sari Alam | 11.619736 |
| Gereja Perawan Maria Tak Berdosa Surabaya | 11.315672 |
| Babakan Siliwangi City Forest Path Bandung | 11.228282 |
| The Lost World Castle | 10.896621 |
| Wisata Lereng Kelir | 10.821424 |

### Kelebihan dan Kekurangan

- **Content-Based Filtering**

  - ✅ **Kelebihan**: Tidak bergantung pada data pengguna lain, cocok untuk pengguna baru (cold-start friendly jika item sudah kaya informasi), dan hasil rekomendasi dapat dijelaskan secara logis.
  - ❌ **Kekurangan**: Rekomendasi bisa menjadi terlalu sempit karena hanya berdasarkan riwayat pengguna sendiri (sering terjadi over-specialization).

- **Collaborative Filtering**

  - ✅ **Kelebihan**:
    - Dapat menangkap preferensi laten atau tersembunyi yang tidak terlihat secara eksplisit dari data konten maupun rating.
    - Lebih akurat dan scalable dalam memproses data interaksi yang besar dan sparsity tinggi.
    - Dapat memberikan prediksi rating bahkan untuk user-item pair yang belum pernah berinteraksi sebelumnya.
  - ❌ **Kekurangan**:
    - Tidak dapat memberikan justifikasi yang jelas terhadap rekomendasi karena bekerja berdasarkan representasi matematis dalam ruang laten.
    - Tetap menghadapi masalah cold-start, terutama jika ada pengguna atau tempat wisata baru tanpa interaksi sebelumnya.
    - Membutuhkan proses training dan tuning parameter yang tepat agar performa optimal, berbeda dengan pendekatan sederhana berbasis similarity.

---

## Evaluasi
Evaluasi dilakukan untuk mengukur seberapa baik sistem rekomendasi memberikan hasil yang relevan dan akurat terhadap preferensi pengguna. Karena proyek ini menggunakan dua pendekatan — **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)** — maka metode evaluasi juga disesuaikan dengan masing-masing pendekatan.

### Evaluasi Content-Based Filtering (CBF)
Evaluasi untuk pendekatan CBF dilakukan menggunakan metrik **Precision@K**, yang mengukur proporsi destinasi yang relevan di antara top-K rekomendasi yang diberikan sistem. Dalam proyek ini, relevansi ditentukan secara manual berdasarkan kecocokan atribut kategori dan kota dari tempat wisata.

$$
\text{Precision@K} = \frac{\text{Jumlah item relevan dalam rekomendasi top-K}}{K}
$$

#### Studi Kasus Evaluasi
Sebagai contoh, pengguna mencari rekomendasi berdasarkan destinasi:
- **Destinasi Asal**: *Pasar Seni*
- **Kategori**: *pusat belanja*
- **Kota**: *jakarta*

Dari hasil top-10 rekomendasi, berikut beberapa pengamatan:

| Recommended Destination | Category | Category | Description Similarity | Category Score | City Score | Total Score |
| ------ | ------ |------ | ------ | ------ | ------ | ------ |
| Pasar Tanah Abang | pusat belanja | jakarta | 0.2060 | 1 | 1 | 0.5236 |
| Pasar Taman Puring | pusat belanja | jakarta | 0.1362 | 1 | 1 | 0.4817 |
| Mall Thamrin City | pusat belanja | jakarta | 0.0530 | 1 | 1 | 0.4318 |
| Plaza Indonesia | pusat belanja | jakarta | 0.0525 | 1 | 1 | 0.4315 |
| Pecinan Glodok | pusat belanja | jakarta | 0.0429 | 1 | 1 | 0.4257 |
| Pasar Petak Sembilan | pusat belanja | jakarta | 0.0317 | 1 | 1 | 0.4190 |
| Kawasan Kuliner BSM | pusat belanja | jakarta | 0.0216 | 1 | 1 | 0.4130 |
| Wisata Kuliner Pecenongan | pusat belanja | jakarta | 0.0147 | 1 | 1 | 0.4088 |
| Pasar Baru | pusat belanja | bandung | 0.2566 | 1 | 0 | 0.4040 |
| Grand Indonesia Mall | pusat belanja | jakarta | 0.0065 | 1 | 1 | 0.4039 |

Dari hasil top-10 rekomendasi, berikut hasil evaluasi berdasarkan Precision@10:

| Kriteria Evaluasi             | Jumlah Relevan | Precision@10 |
|------------------------------|----------------|--------------|
| **Kategori**           | 10 dari 10     | 1.0          |
| **Kota**               | 9 dari 10      | 0.9          |
| **Kategori & Kota**    | 9 dari 10      | 0.9          |

#### Analisis Evaluasi
- **Kekuatan CBF**: mampu menangkap relevansi konten. Hasil seperti *Pasar Tanah Abang* dan *Pasar Taman Puring* memang relevan sebagai alternatif dari *Pasar Seni*.
- **Poin Plus**: meskipun deskripsi similarity rendah, kehadiran atribut *kategori* dan *kota* yang sama menjaga relevansi.
- **Potensi Kekurangan**: karena sistem ini hanya melihat konten, ia tidak mempertimbangkan popularitas atau rating destinasi oleh pengguna lain. Misalnya, sistem tidak tahu apakah *Pasar Baru* disukai pengguna serupa karena dia hanya melihat deskripsi dan atribut tempat.

#### Kesimpulan
Evaluasi menunjukkan bahwa CBF dapat memberikan rekomendasi yang masuk akal berdasarkan atribut tempat. Namun, keterbatasannya adalah pada aspek “apa yang disukai pengguna lain”, yang tidak bisa ditangkap oleh pendekatan ini — inilah yang dilengkapi oleh **Collaborative Filtering**.

### Evaluasi Collaborative Filtering (Matrix Factorization - SVD)
Untuk CF, digunakan pendekatan **Matrix Factorization** yang dievaluasi menggunakan dua metrik error prediktif:

#### **Mean Absolute Error (MAE)**

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

- Mengukur rata-rata selisih absolut antara rating aktual dan prediksi.
- Semakin rendah MAE, semakin akurat prediksi sistem.

#### **Root Mean Squared Error (RMSE)**

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

- Memberikan penalti lebih besar untuk prediksi yang sangat meleset.
- Cocok untuk mendeteksi prediksi outlier.

#### **Hasil Evaluasi CF**

| Metrik | Nilai |
|--------|-------|
| **RMSE**   | 0.1073 |
| **MAE**    | 0.0848 |

> Hasil evaluasi menunjukkan bahwa model Collaborative Filtering dengan pendekatan Matrix Factorization telah berhasil membangun prediksi rating berdasarkan data interaksi pengguna.

---

## Kesimpulan
Sistem rekomendasi destinasi wisata yang dikembangkan dalam proyek ini menggabungkan dua pendekatan utama — **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)** — untuk memberikan saran destinasi yang relevan dan personal kepada pengguna. Evaluasi terhadap kedua metode ini tidak hanya dilakukan dari sisi teknis, tetapi juga ditinjau ulang untuk memastikan bahwa solusi yang dibangun memberikan dampak nyata terhadap kebutuhan bisnis yang telah dirumuskan sebelumnya.

### Relevansi terhadap Problem Statement dan Goals
1. **CBF** berhasil menjawab tantangan dalam merekomendasikan destinasi berdasarkan karakteristik kontennya. Dengan menggunakan teknik TF-IDF dan cosine similarity, sistem mampu mengidentifikasi destinasi serupa berdasarkan deskripsi, kategori, dan kota. Evaluasi menggunakan **precision\@10 secara manual** menunjukkan performa yang sangat baik:

   * Precision\@10 (Kategori): 1.0
   * Precision\@10 (Kota): 0.9
   * Precision\@10 (Kategori & Kota): 0.9

     Hal ini menunjukkan bahwa sistem mampu memberikan hasil yang sangat relevan secara atribut — sesuai dengan tujuan untuk menghadirkan rekomendasi yang mirip secara konten.

2. **CF dengan Matrix Factorization** digunakan untuk menangkap preferensi laten antar pengguna dan berhasil menjawab tantangan dalam merekomendasikan destinasi berdasarkan riwayat dan penilaian dari user/pengguna. Evaluasi prediktif menunjukkan nilai **RMSE = 1.4485** dan **MAE = 1.2247** dapat memberikan estimasi preferensi pengguna terhadap destinasi yang belum mereka eksplorasi.

### Dampak Solusi terhadap Tujuan Bisnis
* Dengan CBF, sistem dapat melayani pengguna baru (cold-start) dengan cukup efektif, karena hanya bergantung pada konten tempat wisata.
* CF berkontribusi dalam meningkatkan keterlibatan pengguna jangka panjang karena mampu menyajikan rekomendasi yang personal dan lebih beragam, berdasarkan kesamaan preferensi dengan pengguna lain.
* Kedua pendekatan ini secara komplementer menjawab *problem statements* yang diajukan dan mendukung pencapaian goals, yaitu membangun sistem rekomendasi yang informatif, relevan, dan mendukung proses pengambilan keputusan bagi pengguna.

---
