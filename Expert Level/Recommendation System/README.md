# Laporan Proyek Machine Learning - Nugroho Adi Wirapratama

## Project Overview

Di era digital saat ini, informasi mengenai tempat wisata dapat dengan mudah diakses secara online. Namun, banyaknya pilihan sering kali membuat pengguna kebingungan dalam menentukan destinasi yang sesuai dengan preferensi mereka. Untuk itu, sistem rekomendasi destinasi wisata menjadi sangat relevan dan dibutuhkan sebagai solusi dalam membantu pengguna menemukan tempat yang paling sesuai dengan minat dan kebutuhan mereka.

Proyek ini bertujuan untuk membangun sistem rekomendasi wisata berbasis Content-Based Filtering (CBF) dan Collaborative Filtering (CF) dengan pendekatan machine learning, yang mampu memberikan rekomendasi secara personal kepada pengguna berdasarkan deskripsi tempat, kategori, lokasi, maupun interaksi pengguna sebelumnya seperti rating.

**Mengapa proyek ini harus diselesaikan ?**

Pentingnya proyek ini diselesaikan tidak hanya terletak pada nilai praktisnya dalam meningkatkan pengalaman pengguna saat merencanakan perjalanan, tetapi juga dalam mengurangi waktu pencarian dan mendukung promosi destinasi lokal yang relevan. Dengan adanya sistem rekomendasi, pengguna akan lebih mudah menemukan tempat-tempat yang mungkin belum populer, namun sesuai dengan ketertarikan mereka.
  
**Referensi:**
- Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer. [Link](https://www.researchgate.net/publication/227268858_Recommender_Systems_Handbook)

**Dataset :** https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination

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

## Data Understanding
Terdapat 4 dataset .csv yang tersedia yaitu
- tourism_ with _id.csv : berisi informasi tentang objek wisata di 5 kota besar di Indonesia dengan total sekitar 400 tempat wisata.
- user.csv : berisi data pengguna dummy untuk membuat fitur rekomendasi berdasarkan pengguna.
- tourism_rating.csv : berisi 3 kolom, yaitu pengguna, tempat wisata, dan rating yang diberikan, digunakan untuk membuat sistem rekomendasi berdasarkan rating.
- package_tourism.csv : berisi rekomendasi tempat wisata terdekat berdasarkan waktu, biaya, dan rating.

File .csv yang digunakan untuk pembuatan sistem rekomendasi ini hanya **tourism_ with _id.csv** dan **tourism_rating.csv**. Kedua file tersebut dilakukan merge dataset.

### Variabel pada dataset yang sudah di merge:
- User_Id : user_id.
- Place_Id : id dari Place_Name.
- Place_Name : nama tempat destinasi.
- Description : teks deskripsi tentang destinasi.
- Category : kategori dari destinasi.
- City : lokasi kota dari destinasi.
- Price : harga tiket masuk destinasi.
- Rating : rating yang diberikan User_Id terhadap destinasi.

| User_Id | Place_Id | Place_Name | Description | Category | City | Price | Rating |
| ------ | ------ |------ | ------ | ------ | ------ | ------ | ------ |
| 1 | 179 | Candi Ratu Boko | Situs Ratu Baka atau Candi Boko (Hanacaraka:ꦕꦤ... | Budaya | Yogyakarta | 75000 | 4.6 |
| 1 | 344 | Pantai Marina | Pantai Marina (bahasa Jawa: ꦥꦱꦶꦱꦶꦂ​ꦩꦫꦶꦤ, trans... | Bahari | Semarang | 3000 | 4.1 |
| 1 | 5   | Atlantis Water Adventure | Atlantis Water Adventure atau dikenal dengan A... | Taman Hiburan | Jakarta | 94000 | 4.5 |
| 1 | 373 | Museum Kereta Ambarawa | Museum Kereta Api Ambarawa (bahasa Inggris: In... | Budaya | Semarang | 10000 | 4.5 |
| 1 | 101 | Kampung Wisata Sosro Menduran | Kampung wisata Sosromenduran merupakan kampung... | Budaya | Yogyakarta | 0 | 4.0 |

### Dataset information
![image](https://github.com/user-attachments/assets/9a399bbc-d2a2-40a2-952f-a39a436feafb)

Dataset memiliki 10000 baris dan 8 kolom. Untuk tipe data pada dataset sudah sesuai.

### NaN Value
![image](https://github.com/user-attachments/assets/9a399bbc-d2a2-40a2-952f-a39a436feafb)

Tidak ada baris yang memiliki NaN value

### Data Duplikat
![image](https://github.com/user-attachments/assets/2b16ddab-146a-471e-8203-161f5c41b000)

Terdapat 403 data duplikat, yang merupakan 4.03% dari keseluruhan data.

### Top 10 Places by Average Rating
![image](https://github.com/user-attachments/assets/6bd76626-2665-466c-ad29-110c07279672)

Wisata Kuliner Pecenongan menduduki peringkat pertama dengan rating tertinggi, diikuti oleh Desa Wisata Sungai Code Jogja Kota dan Freedom Library.

### Scatter Plot: Price vs Rating
![image](https://github.com/user-attachments/assets/f3a228e3-0c5f-482b-8009-17b36d55def3)

Dari scatter plot, terlihat bahwa meskipun harga bervariasi, sebagian besar destinasi memiliki rating tinggi (di atas 4), namun tidak ada pola yang jelas antara harga dan rating.

![image](https://github.com/user-attachments/assets/d5eeeece-2433-486f-a330-7f70aaea2e8f)

Dalam analisis regresi, hasil menunjukkan bahwa koefisien untuk harga sangat kecil (7.296e-08), dan p-value sebesar 0.628 menunjukkan bahwa harga tidak berpengaruh signifikan terhadap rating destinasi wisata. Nilai R-squared yang sangat rendah (0.001) juga menegaskan bahwa harga hanya menjelaskan sedikit sekali variasi dalam rating.

**Harga tidak menjadi faktor yang signifikan dalam menentukan rating destinasi wisata.**

### Barplot Distribusi Rating
![image](https://github.com/user-attachments/assets/200577e3-8c0d-4fbc-9c56-f1b93f43f73e)

Sebagian besar rating yang diterima adalah rating 4, dengan jumlah yang sangat dominan hampir mencapai 7.000. Sementara itu, rating 5 juga mendapatkan jumlah yang cukup besar, meskipun jauh lebih sedikit dibandingkan rating 4. Di sisi lain, rating 3 hampir tidak terlihat di grafik, menandakan bahwa sedikit sekali tempat yang menerima penilaian di sekitar nilai ini.

## Data Preparation
Ada beberapa tahapan untuk data preparation sebelum memulai untuk membuat sistem rekomendasi. Sebelum dilakukannya data preparation, dataset memiliki **jumlah baris 10000** dan **jumlah kolom 8**.

![image](https://github.com/user-attachments/assets/0cbf8607-fefe-447b-ad16-e9678abffc26)

### 1. Menghapus kolom yang tidak relevan

Kolom yang dihapus adalah Price. Berdasarkan data understanding pada bagian **Scatter Plot: Price vs Rating**, kolom Price tidak ada korelasi terhadap rating.

### 2. Drop/delete duplikat value

Terdapat 403 baris duplikat, yang merepresentasikan sekitar 4.03% dari keseluruhan data. Karena baris-baris ini sepenuhnya identik, maka dihapus untuk menghindari bias pada hasil pemodelan.

### 3. Processing text

Processing text dilakukan pada kolom **Description**, **City**, dan **Category**. Tahapan processing text yang dilakukan adalah sebagai berikut:
- Lowercasing
- Penghapusan tanda baca dan angka
- Penghapusan stopwords Bahasa Indonesia
- Stemming menggunakan Sastrawi

Tujuan dari preprocessing ini adalah untuk menyederhanakan teks dan meningkatkan kualitas perhitungan kemiripan antar destinasi wisata.

![image](https://github.com/user-attachments/assets/8a76112a-bb9b-4a90-b0ce-7b7e790f54be)

Setelah dilakukan beberapa tahapan preparation, jumlah baris dan kolom pada dataset berjumlah **9597 baris** dan **7 kolom**

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

Alih-alih menghitung kemiripan antar pengguna secara langsung, pendekatan ini membentuk user-item matrix dan menerapkannya pada model Matrix Factorization (misalnya menggunakan algoritma SVD). Model ini akan mempelajari representasi laten dari pengguna dan destinasi untuk memprediksi rating yang hilang dan merekomendasikan tempat yang paling relevan.

Pembentukan user-item matrix
```python
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_cf[['User_Id', 'Place_Name', 'Rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)
```

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

## Evaluasi
Evaluasi dilakukan untuk mengukur seberapa baik sistem rekomendasi memberikan hasil yang relevan dan akurat terhadap preferensi pengguna. Karena proyek ini menggunakan dua pendekatan — **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)** — maka metode evaluasi juga disesuaikan dengan masing-masing pendekatan.

### Evaluasi Content-Based Filtering (CBF)
Evaluasi sistem CBF dalam proyek ini dilakukan secara **manual** berdasarkan relevansi dari atribut konten (kategori, kota, dan deskripsi) terhadap tempat wisata yang direkomendasikan. Sistem menghitung skor total berdasarkan:
- **Deskripsi** (60%) – menggunakan TF-IDF dan cosine similarity  
- **Kategori** (25%) – kesamaan kategori  
- **Kota** (15%) – kesamaan lokasi  

Rekomendasi dievaluasi berdasarkan apakah hasil yang muncul memiliki atribut yang mirip dengan tempat wisata asal. Evaluasi dilakukan dengan:
- **Visual inspection** terhadap hasil top-N rekomendasi
- **Pemeriksaan kesesuaian** berdasarkan kesamaan deskripsi, kategori, dan kota

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

> Terlihat bahwa hampir seluruh hasil memiliki **kategori yang sama** (*pusat belanja*) dan berada di **lokasi yang sama** (Jakarta), dengan deskripsi yang mengandung kesamaan tematik seperti “pasar”, “belanja”, “kawasan seni”, dan “kolektor”.

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

> Hasil evaluasi menunjukkan bahwa model CF dengan Matrix Factorization memiliki akurasi prediksi yang baik, dengan error yang rendah.

**Dengan nilai RMSE sebesar 0.1102 dan MAE 0.0867, model memiliki kemampuan prediksi yang cukup tinggi. Artinya, sistem mampu memperkirakan rating pengguna terhadap destinasi wisata dengan tingkat kesalahan yang sangat kecil.**
