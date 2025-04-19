# Laporan Proyek Machine Learning - Nugroho Adi Wirapratama

## Domain Proyek

Dalam dunia investasi saham, kemampuan untuk memprediksi harga saham secara akurat menjadi hal yang sangat penting, baik bagi investor individu maupun institusi keuangan. Prediksi harga saham yang tepat dapat membantu pengambilan keputusan investasi, mengurangi risiko kerugian, dan meningkatkan potensi keuntungan.

PT Bank Rakyat Indonesia (Persero) Tbk atau BBRI merupakan salah satu perusahaan BUMN terbesar di Indonesia yang sahamnya memiliki kapitalisasi pasar tinggi dan likuiditas yang sangat baik. Saham BBRI juga termasuk dalam indeks LQ45 dan IDX30, menjadikannya salah satu saham yang paling aktif diperdagangkan dan banyak diminati oleh investor.

**Mengapa masalah ini harus diselesaikan? Dan bagaimana menyelesaikannya?**

Seperti diketahui bahwa pergerakan harga saham pada umumnya sangat berfluktuatif, hal itu sejalan dengan risk yang diterima oleh Investor. Masalah ini dapat diselesaikan dengan pendekatan machine learning yang mampu mempelajari pola dari data historis. Dengan model yang tepat, prediksi harga saham bisa menjadi lebih akurat dan membantu investor dalam mengambil keputusan yang lebih rasional.
  
**Referensi:**
- Dataset didapat dari hasil scraping website Yahoo Finannce [BBRI.JK](https://finance.yahoo.com/quote/BBRI.JK/)
- [Paper](https://www.researchgate.net/publication/363040938_Prediksi_Data_Time_Series_Saham_Bank_BRI_Dengan_Mesin_Belajar_LSTM_Long_ShortTerm_Memory/) 

## Business Understanding
Pasar saham merupakan salah satu instrumen investasi dengan volatilitas yang tinggi, sehingga memerlukan pendekatan analitik yang kuat untuk memahami dan memprediksi pergerakannya. Salah satu saham yang sangat aktif diperdagangkan di Bursa Efek Indonesia adalah saham PT Bank Rakyat Indonesia (Persero) Tbk (BBRI). Karena pergerakan harganya yang fluktuatif dan menarik bagi investor, saham ini menjadi objek yang relevan untuk diteliti lebih lanjut dalam konteks prediksi harga.

Pergerakan harga saham merupakan data time series yang sangat dipengaruhi oleh pola masa lalu. Untuk itu, diperlukan model yang mampu mengenali ketergantungan jangka panjang dari data historis. Salah satu algoritma yang efektif untuk tugas ini adalah Long Short-Term Memory (LSTM), yaitu jenis Recurrent Neural Network (RNN) yang dirancang khusus untuk mengatasi kelemahan RNN biasa dalam mengingat informasi jangka panjang.

### Problem Statements
- Bagaimana cara memprediksi harga saham BBRI berdasarkan data historis?
- Bagaimana cara memberikan insight arah tren pergerakan harga saham BBRI untuk mendukung keputusan investasi seperti pembelian atau penjualan saham?

### Goals
- Menghasilkan prediksi harga saham BBRI menggunakan data historis sebagai dasar pengambilan keputusan investasi.
- Memberikan insight mengenai tren harga saham BBRI di masa depan yang dapat membantu investor dalam mempertimbangkan waktu yang tepat untuk membeli atau menjual saham.

### Solution statements
1. Membangun Model LSTM
   - Menggunakan arsitektur LSTM untuk memproses data historis harga saham dalam bentuk time series.
   - Menggunakan pendekatan sliding window/timestep untuk membentuk input sekuensial.

2. Hyperparameter Tuning
   - Melakukan eksplorasi terhadap nilai-nilai optimal dari beberapa hyperparameter penting, antara lain:
     - Epoch
     - Timestep
     - Jumlah units neuron
     - Jumlah nilai dropout
   - Proses tuning dilakukan secara iteratif atau dengan teknik pencarian seperti Random Search.

3. Evaluasi dan Visualisasi
   - Menggunakan RMSE, MAE, dan R² Score sebagai metrik evaluasi model.
   - Membandingkan grafik hasil prediksi dengan data aktual untuk menilai kualitas prediksi secara visual.

## Data Understanding
Dataset yang digunakan merupakan data historis dari saham BBRI selama 3 tahun (1 Januari 2022 - 31 Desember 2024). Baris data yang berhasil didapatkan sebanyak 728 baris dan 6 kolom. 

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

### Dataset information
![image](https://github.com/user-attachments/assets/315b58d3-5013-4161-9e5d-db5925e88bca)

Tipe data pada semua kolom masih berupa object/string.

### NaN value

![Jumlah NaN](https://github.com/user-attachments/assets/15d723fe-1e23-4f34-b8d6-3d01880b6ce9) | ![Distribusi](https://github.com/user-attachments/assets/93b858e7-09d7-4cfd-97e3-4089aea3d142) | ![Boxplot](https://github.com/user-attachments/assets/ada30185-352a-4fd6-b1b8-f7c0707ed63b) |
|:--:|:--:|:--:|

- Gambar pertama menunjukkan dataset memiliki null value pada kolom Low, Close, dan Volume sebanyak 6 baris. 
- Terlihat bahwa value hanya terisi pada kolom Open. Jika kita lihat pada kolom lainnya, value dari Open pun tidak berada di rentang puluhan/ratusan.

**Setelah melihat kembali pada data di website Yahoo Finance kembali, NaN value tersebut adalah besaran nilai dividen yang dibagikan oleh perusahaan kepada investor pada hari itu.**

### Data duplikat

![image](https://github.com/user-attachments/assets/fe929bfc-42ab-4433-84f6-9978631690eb)

Dataset tidak memiliki data yang duplikat.

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
1. **Mengubah Tipe Data**

![image](https://github.com/user-attachments/assets/da8989f2-d572-4554-9873-7f01a9b2cf56)

Dilakukan convert tipe data dari yang semua nya object/string ke tipe data yang sesuai
- Date -> **datetime**
- Open -> **float**
- High -> **float**
- Low -> **float**
- Close -> **float**
- Volume -> **float**

2. **Drop/delete NaN value**

Dikarenakan NaN value disebabkan oleh pengisian besaran dividen yang diberikan perusahaan oleh investor, baris tersebut dilakukan drop/delete.

![image](https://github.com/user-attachments/assets/c54f76cc-c64a-4748-9083-8b6d72fe7ed3)

3. **Date sebagai Index**

Untuk mempermudah pemrosesan data time series, kolom tanggal (Date) dijadikan sebagai index. Hal ini dilakukan agar setiap data harga saham dapat diakses dan dianalisis berdasarkan waktu secara kronologis.

4. **Normalisasi Data**

Arsitekur LSTM sangat sensitif terhadap skala data. Oleh karena itu, dilakukan normalisasi menggunakan MinMaxScaler untuk mengubah skala nilai ke dalam rentang [0, 1]. Tujuannya adalah agar proses pembelajaran pada model LSTM menjadi lebih stabil dan cepat konvergen.

5. **Penentuan Time Step**

Dalam konteks time series forecasting, digunakan pendekatan sliding window dengan time step sebanyak 5, yang berarti model akan menggunakan lima data sebelumnya untuk memprediksi harga pada waktu berikutnya. Pemilihan time step ini merupakan salah satu hyperparameter penting dalam modeling time series.

6. **Split Dataset**

Data kemudian dibagi menjadi dua bagian, yaitu training set (80%) dan testing set (20%). Dataset yang telah terbentuk kemudian di-reshape menjadi format tiga dimensi [samples, time steps, features], yaitu format yang dibutuhkan oleh model LSTM dalam proses pelatihan.

## Modeling
Setelah melakukan analisis dan data prepration, selanjutnya dilakukan proses modeling. pemodelan dilakukan dengan menggunakan algoritma Long Short-Term Memory (LSTM), yang dikenal efektif dalam mengolah data sekuensial seperti data time series saham.

![image](https://github.com/user-attachments/assets/bcadf48e-f9fa-429e-b6f7-5217137bbdcc)

Setelah dilakukan EDA (Exploratory Data Analysis) terlihat bahwa pergerakan harga saham BBRI memiliki rentang yang cukup fluktuatif, sehingga arsitektur memiliki kemampuan yang optimal untuk digunakan dikarenakan LSTM dapat memahami pola untuk jangka panjang dan kemampuan untuk mengatasan pergerakan harga yang kompleks.

1. **Arsitektur dan Pengembangan Model**

Model awal dibangun menggunakan pendekatan stacked LSTM, yang terdiri dari dua lapisan LSTM berurutan. Setiap lapisan diikuti oleh layer Dropout untuk mencegah overfitting. Arsitektur ini disusun sebagai berikut:
  - **Lapisan pertama**: LSTM dengan return_sequences=True untuk mengizinkan data mengalir ke layer LSTM berikutnya.
  - **Lapisan kedua**: LSTM dengan return_sequences=False karena ini adalah lapisan terakhir yang memproses urutan.
  - **Lapisan output**: Dense dengan satu neuron sebagai output regresi.
Fungsi aktivasi default digunakan (tanh untuk LSTM dan linear untuk output). Optimizer yang digunakan adalah Adam, karena umumnya stabil dan cepat konvergen dalam training LSTM.

2. **Hyperparameter Tuning dengan Keras Tuner**

Untuk meningkatkan performa model, dilakukan tuning hyperparameter otomatis menggunakan modul RandomSearch dari Keras Tuner. Parameter-parameter yang diuji meliputi:
  - Jumlah neuron pada masing-masing lapisan LSTM (units, units_2)
  - Tingkat dropout untuk masing-masing lapisan (dropout, dropout_2)
  - Jumlah epoch dan batch size tetap konstan selama tuning
Proses tuning dieksekusi sebanyak lima kali percobaan (trials) dengan satu eksekusi per trial, untuk mendapatkan kombinasi hyperparameter terbaik berdasarkan nilai val_loss terendah.
```
tuner = RandomSearch(build_model,
                     objective='val_loss',
                     max_trials=5,
                     executions_per_trial=1,
                     directory='my_dir',
                     project_name='lstm_tuning')
```
3. **Best Parameter**

Setelah proses tuning selesai, model terbaik dibangun kembali dengan konfigurasi yang memberikan performa validasi terbaik. Adapun konfigurasi tersebut adalah:
  - Lapisan LSTM pertama: 150 unit neuron, return_sequences=True, dropout 0.4
  - Lapisan LSTM kedua: 200 unit neuron, dropout 0.3
  - Output layer: Dense (1 unit)
  - Optimizer: Adam
  - Loss function: Mean Squared Error (MSE)
  - Epoch: 20
  - Batch size: 32
Model ini kemudian dilatih kembali pada data pelatihan dan divalidasi menggunakan data testing. Proses pelatihan menggunakan 20 epoch dan menunjukkan stabilitas serta penurunan loss yang signifikan pada data validasi

## Evaluation
Untuk mengevaluasi performa model LSTM dalam memprediksi harga saham BBRI, digunakan beberapa metrik regresi berikut:
- Root Mean Squared Error (RMSE)
RMSE digunakan untuk mengukur jarak antara nilai prediksi dengan nilai aktual dalam satuan aslinya (harga saham). RMSE memberikan penalti yang lebih besar terhadap kesalahan prediksi yang besar.

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

- Mean Absolute Percentage Error (MAPE)
MAPE digunakan untuk mengukur kesalahan prediksi dalam bentuk persentase, yang sangat berguna untuk interpretasi karena tidak bergantung pada satuan. Semakin kecil nilai MAPE, semakin akurat prediksi model.

$$
MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

- R² Score (Koefisien Determinasi)
R² menunjukkan proporsi variasi nilai aktual yang dapat dijelaskan oleh model. Nilai R² mendekati 1 menandakan model yang sangat baik.

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

Parameter terbaik yang diperoleh dari proses tuning kemudian diuji pada data testing.

### ⚙️ Hasil Tuning dan Konfigurasi Optimal
- Timestep: 5
- Units_1: 150
- Units_2: 200
- Dropout_1: 0.4
- Dropout_2: 0.3  
- Optimizer: Adam
- Epoch: 20
- Batch size: 32

Konfigurasi ini memberikan hasil terbaik dalam hal keseimbangan antara akurasi dan stabilitas model.

### Evaluasi Model Forecasting Data Test
![image](https://github.com/user-attachments/assets/a201f8f1-6e73-4861-83da-5bda51456a85)

Model LSTM mampu mengikuti pola pergerakan harga saham dengan cukup baik. Garis prediksi (merah) secara umum mengikuti pergerakan harga aktual (biru), terutama dalam mencerminkan arah tren naik dan turun. Hal ini memperlihatkan bahwa model dapat memberikan gambaran umum mengenai kecenderungan pergerakan harga saham di masa depan, yang berguna sebagai bahan pertimbangan dalam pengambilan keputusan investasi.

*🔍 Catatan:*
- Fluktuasi tajam atau lonjakan ekstrem agak sulit ditangkap secara presisi, namun ini umum dalam data saham.
- Secara keseluruhan, model sudah cukup representatif untuk digunakan dalam konteks analisis dan forecasting.

### ✅ Skor Evaluasi Metrik Model
- RMSE: 102.58155180242858
- MAPE: 1.5988376073752097%
- R² Score: 0.7682263497209625

**Nilai MAPE yang rendah menunjukkan bahwa model memiliki tingkat akurasi tinggi dalam konteks prediksi harga saham. RMSE yang cukup kecil dan R² yang mendekati 1 menunjukkan bahwa model memiliki kemampuan generalisasi yang baik.**

### Evaluasi Visual: Loss Function
![image](https://github.com/user-attachments/assets/1f15541b-31b3-4807-a7cd-44f7da62a924)

**Grafik menunjukkan bahwa model mengalami konvergensi yang baik. Baik training loss maupun validation loss menurun secara konsisten dan tetap rendah hingga akhir epoch, tanpa indikasi overfitting. Hal ini memperkuat bahwa model dapat melakukan generalisasi dengan baik pada data baru.**

## Kesimpulan Model
Model LSTM yang telah dibangun dan dituning mampu memberikan estimasi harga saham BBRI yang cukup akurat berdasarkan data historis. Melalui proses evaluasi kuantitatif (RMSE, MAPE, dan R² Score) dan visualisasi hasil prediksi, model menunjukkan kemampuan dalam mengenali pola pergerakan harga dan mengidentifikasi arah tren di masa depan.

Prediksi yang dihasilkan dari model ini dapat memberikan insight bagi investor, terutama dalam **memahami kecenderungan naik atau turunnya harga saham**. Informasi ini dapat dimanfaatkan untuk mendukung pengambilan keputusan investasi yang lebih rasional, seperti kapan waktu yang tepat untuk mempertimbangkan aksi beli atau jual saham.

Model ini **tidak bertujuan untuk menentukan harga spesifik sebagai acuan jual atau beli**, namun lebih sebagai **alat bantu analitik dalam melihat tren pasar secara umum.**

Dengan akurasi dan stabilitas yang dimiliki, model LSTM ini memiliki potensi untuk diintegrasikan dalam sistem pendukung keputusan investasi, ataupun sebagai bagian dari dashboard analitik bagi analis keuangan dan investor.
