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
PT Bank Rakyat Indonesia (Persero) Tbk (BBRI) merupakan salah satu emiten terbesar di Bursa Efek Indonesia (BEI), dan pergerakan harganya mencerminkan sentimen pasar secara luas. Namun, perubahan harga yang cepat dan tidak menentu memerlukan pendekatan khusus agar prediksi dapat dilakukan secara efektif.

Salah satu pendekatan yang digunakan untuk memahami pola data time series seperti harga saham adalah melalui metode machine learning, khususnya model berbasis Recurrent Neural Network (RNN) seperti Long Short-Term Memory (LSTM). Model ini memiliki kemampuan mengingat informasi jangka panjang, yang membuatnya cocok untuk memproses dan menganalisis data historis saham.

### Problem Statements
- Algoritma apa yang paling sesuai digunakan untuk data time series dalam konteks pasar saham Indonesia?
- Bagaimana evaluasi performa model agar dapat diandalkan dalam pengambilan keputusan investasi?

### Goals
- Mengembangkan sistem prediksi harga saham BBRI dengan menggunakan pendekatan machine learning berbasis time series.
- Mengeksplorasi dan membandingkan performa satu atau lebih model dalam memprediksi harga saham.
- Menentukan metrik evaluasi yang tepat untuk mengukur akurasi prediksi, serta melakukan analisis terhadap performa model.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

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

