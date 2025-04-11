# Laporan Proyek Machine Learning - Nugroho Adi Wirapratama

## Domain Proyek

Dalam dunia investasi saham, kemampuan untuk memprediksi harga saham secara akurat menjadi hal yang sangat penting, baik bagi investor individu maupun institusi keuangan. Prediksi harga saham yang tepat dapat membantu pengambilan keputusan investasi, mengurangi risiko kerugian, dan meningkatkan potensi keuntungan.

PT Bank Rakyat Indonesia (Persero) Tbk atau BBRI merupakan salah satu perusahaan BUMN terbesar di Indonesia yang sahamnya memiliki kapitalisasi pasar tinggi dan likuiditas yang sangat baik. Saham BBRI juga termasuk dalam indeks LQ45 dan IDX30, menjadikannya salah satu saham yang paling aktif diperdagangkan dan banyak diminati oleh investor.

**Mengapa masalah ini harus diselesaikan? Dan bagaimana menyelesaikannya?**

Seperti diketahui bahwa pergerakan harga saham pada umumnya sangat berfluktuatif, hal itu sejalan dengan risk yang diterima oleh Investor. Masalah ini dapat diselesaikan dengan pendekatan machine learning yang mampu mempelajari pola dari data historis. Dengan model yang tepat, prediksi harga saham bisa menjadi lebih akurat dan membantu investor dalam mengambil keputusan yang lebih rasional.
  
**Referensi:**
- Dataset didapat dari hasil scraping website Yahoo Finannce [BBRI.JK](https://finance.yahoo.com/quote/BBRI.JK/)
- [Paper](https://scholar.google.com/) 

## Business Understanding
Pasar saham merupakan salah satu instrumen investasi dengan volatilitas yang tinggi, sehingga memerlukan pendekatan analitik yang kuat untuk memahami dan memprediksi pergerakannya. Salah satu saham yang sangat aktif diperdagangkan di Bursa Efek Indonesia adalah saham PT Bank Rakyat Indonesia (Persero) Tbk (BBRI). Karena pergerakan harganya yang fluktuatif dan menarik bagi investor, saham ini menjadi objek yang relevan untuk diteliti lebih lanjut dalam konteks prediksi harga.

Pergerakan harga saham merupakan data time series yang sangat dipengaruhi oleh pola masa lalu. Untuk itu, diperlukan model yang mampu mengenali ketergantungan jangka panjang dari data historis. Salah satu algoritma yang efektif untuk tugas ini adalah Long Short-Term Memory (LSTM), yaitu jenis Recurrent Neural Network (RNN) yang dirancang khusus untuk mengatasi kelemahan RNN biasa dalam mengingat informasi jangka panjang.

### Problem Statements
- Bagaimana membangun model LSTM yang optimal untuk memprediksi harga saham BBRI berdasarkan data historis?
- Parameter apa saja yang paling mempengaruhi kinerja model LSTM dalam konteks data time series saham?

### Goals
- Membangun model prediksi harga saham BBRI menggunakan algoritma LSTM.
- Melakukan pencarian nilai optimal dari beberapa hyperparameter utama melalui proses tuning.
- Mengukur kinerja model menggunakan metrik evaluasi seperti Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), dan R² Score.
- Menganalisis hasil prediksi baik secara kuantitatif (nilai error) maupun kualitatif (visualisasi grafik prediksi vs data aktual).

### Solution statements
1. Membangun Model LSTM
   - Menggunakan arsitektur LSTM untuk memproses data historis harga saham dalam bentuk time series.
   - Menggunakan pendekatan sliding window/timestep untuk membentuk input sekuensial.

2. Hyperparameter Tuning
   - Melakukan eksplorasi terhadap nilai-nilai optimal dari beberapa hyperparameter penting, antara lain:
     - Epoch
     - Timestep
     - Jumlah neuron di hidden layer
     - Batch size
     - Learning rate
   - Proses tuning dilakukan secara iteratif atau dengan teknik pencarian seperti Grid Search atau Random Search.

4. Evaluasi dan Visualisasi
   - Menggunakan RMSE, MAE, dan R² Score sebagai metrik evaluasi model.
   - Membandingkan grafik hasil prediksi dengan data aktual untuk menilai kualitas prediksi secara visual.

## Data Understanding
Dataset yang digunakan merupakan data historis dari saham BBRI selama 3 tahun (1 Januari 2022 - 31 Desember 2024). Baris data yang berhasil didapatkan sebanyak 728 baris. 

Pengambilan data dilakukan dengan cara scraping pada platform [Yahoo Finance](https://finance.yahoo.com/quote/BBRI.JK/).

### Variabel yang digunakan pada forecasting harga saham BBRI adalah:
- Date : Tanggal perdagangan saham yang tercatat, yaitu hari di mana transaksi saham dilakukan di pasar.
- Open : Harga pertama yang tercatat pada saat pasar dibuka pada hari tersebut.
- High : Harga tertinggi yang tercatat selama sesi perdagangan pada hari itu.
- Low : Harga terendah yang tercatat selama sesi perdagangan pada hari itu.
- Closing : Harga saham pada saat pasar ditutup pada akhir sesi perdagangan.
- Volume : Jumlah total saham yang diperdagangkan pada hari tersebut.

| Date | Open | High | Low | Close | Volume |
| ------ | ------ |------ | ------ | ------ | ------ |
| 2024-12-30 | 4080.0 | 4120.0 | 4070.0 | 4080.0 | 153934700.0 |
| 2024-12-27 | 4100.0 | 4120.0 | 4080.0 | 4100.0 | 143104400.0 |
| 2024-12-24 | 4220.0 | 4250.0 | 4170.0 | 4200.0 | 199536100.0 |
| 2024-12-23 | 4130.0 | 4210.0 | 4110.0 | 4210.0 | 167689800.0 |
| 2024-12-20 | 4070.0 | 4120.0 | 4050.0 | 4060.0 | 252689600.0 |

### Tren (3 years)
![image](https://github.com/user-attachments/assets/03cb0907-172e-4e39-8ce3-aee3d61566b6)
Grafik ini memperlihatkan fluktuasi harga saham BBRI dari awal 2022 hingga akhir 2024. Secara keseluruhan, ada tren kenaikan yang signifikan dari 2022 ke 2023, dengan beberapa fluktuasi tajam. Kemudian, harga saham mulai turun secara drastis setelah 2024.

Moving average 30 hari memberikan gambaran harga saham BBRI cenderung naik dengan lonjakan-lonjakan signifikan, meskipun ada periode penurunan yang cukup tajam. Moving average membantu menghaluskan fluktuasi harian dan memperlihatkan tren jangka panjang yang lebih stabil.

### Tren Per Tahun
![image](https://github.com/user-attachments/assets/3ba5e5e6-ac61-4009-a2fc-8e5b90f0d2bb)
Harga penutupan BBRI cenderung akan naik di awal tahun sampai pertengahan tahun, dan akan kembali naik mendekati akhir tahun.

Tahun 2024 agar berbeda dikarenakan adanya keresahan investor mengenai isu politik dengan pemilihan presiden.

### Rata-rata Harga Saham BBRI Per Bulan
![image](https://github.com/user-attachments/assets/d04b770f-f9d7-4fde-9e9b-b89b44972eb8)
Ada beberapa bulan dengan harga yang lebih tinggi, dan ada beberapa bulan yang harga sahamnya lebih rendah. Hal ini bisa menunjukkan adanya pola musiman atau faktor-faktor eksternal yang mempengaruhi harga saham pada waktu-waktu tertentu.

### Distribusi & Boxplot
| ![Distribusi](https://github.com/user-attachments/assets/58d7fd63-d345-4898-bab7-b076d7fa9381) | ![Boxplot](https://github.com/user-attachments/assets/99fe7fa6-3414-409a-a452-2347e54422c9) |
|:--:|:--:|
- Distribusi Tidak Normal
- Skewness Positif
- Tidak ada outlier

### Return Harian
![image](https://github.com/user-attachments/assets/3c455ad1-eca0-4f43-a7a0-f942793d83a9)
Grafik menunjukkan fluktuasi harian yang cukup besar, dengan banyak lonjakan tajam baik ke atas maupun ke bawah. Beberapa puncak tajam menunjukkan periode dengan perubahan harga yang sangat besar dalam waktu singkat. Ini mengindikasikan bahwa harga saham BBRI mengalami volatilitas yang cukup tinggi di banyak titik waktu.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

