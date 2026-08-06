# Local RAG Assistant

Microsoft Foundry Local kullanılarak geliştirilen, tamamen yerel çalışan bir RAG tabanlı soru-cevap uygulaması.

## Projenin Amacı

Bu projenin amacı, yerel dokümanlardan kullanıcı sorusuyla ilgili bilgileri bulmak ve yalnızca bu bilgilere dayanarak cevap üretmektir.

Uygulama internet bağlantısı olmadan çalışır. Belgeler yerel olarak işlenir, embedding vektörleri SQLite veritabanında saklanır ve cevaplar yerel bir dil modeli tarafından oluşturulur.

## Kullanılan Teknolojiler

- Python
- Microsoft Foundry Local
- SQLite
- Text Embeddings
- Cosine Similarity
- Retrieval-Augmented Generation
- Yerel LLM
- OpenAI uyumlu yerel API

## Sistem Nasıl Çalışır?

Uygulamanın temel akışı şöyledir:

```text
Yerel belgeler
    ↓
Metin parçalarına ayırma
    ↓
Embedding oluşturma
    ↓
SQLite veritabanına kaydetme
    ↓
Kullanıcı sorusunun embedding'ini oluşturma
    ↓
Cosine similarity ile en ilgili parçaları bulma
    ↓
Benzerlik eşiği kontrolü
    ↓
İlgili parçaları yerel LLM'e bağlam olarak gönderme
    ↓
Belgelere dayalı cevap üretme
```

## Proje Özellikleri

- Yerel `.txt` belgelerini okur.
- Belgeleri paragraf tabanlı parçalara ayırır.
- Her parça için embedding üretir.
- Metinleri ve embedding vektörlerini SQLite veritabanında saklar.
- Kullanıcı sorusuna en yakın belge parçalarını cosine similarity ile bulur.
- En ilgili iki belge parçasını cevap üretiminde kullanır.
- Düşük benzerlik puanına sahip soruları filtreler.
- Belgelerde cevabı bulunmayan sorulara cevap üretmez.
- Cevapla birlikte kullanılan kaynak dosyaları ve benzerlik puanlarını gösterir.
- Tüm modeli ve veriyi yerel cihazda çalıştırır.

## Kullanılan Modeller

Embedding modeli:

```text
qwen3-embedding-0.6b
```

Sohbet modeli:

```text
qwen3.5-2b-text
```

## Proje Yapısı

```text
local-rag-assistant/
│
├── data/
│   └── documents/
│       ├── foundry_local_notes.txt
│       ├── rag_notes.txt
│       └── sqlite_notes.txt
│
├── examples/
│   ├── check_database.py
│   ├── database_demo.py
│   ├── embedding_demo.py
│   ├── hello_model.py
│   ├── list_models.py
│   └── main.py
│
├── ingest_documents.py
├── retrieval.py
├── rag_assistant.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Ana Dosyalar

### ingest_documents.py

`data/documents` klasöründeki metin belgelerini okur.

Görevleri:

- Belgeleri bulur.
- Metinleri parçalara ayırır.
- Her parça için embedding oluşturur.
- Verileri `data/rag_database.db` dosyasına kaydeder.

### retrieval.py

Kullanıcı sorusuna en yakın belge parçalarını bulur.

Görevleri:

- Kullanıcı sorusunun embedding'ini oluşturur.
- SQLite veritabanındaki embedding vektörlerini okur.
- Cosine similarity hesaplar.
- En ilgili belge parçalarını sıralar.

### rag_assistant.py

Projenin ana uygulamasıdır.

Görevleri:

- Embedding ve sohbet modellerini yükler.
- Kullanıcıdan soru alır.
- İlgili belge parçalarını bulur.
- Minimum benzerlik eşiğini kontrol eder.
- Belge bağlamını yerel dil modeline gönderir.
- Kaynaklara dayalı cevap üretir.
- Kullanılan kaynakları terminalde gösterir.

## Kurulum

Projeyi klonlayın:

```powershell
git clone <https://github.com/sumeyyekynk7/local-rag-assistant.git>
cd local-rag-assistant
```

Sanal ortam oluşturun:

```powershell
python -m venv .venv
```

Windows PowerShell üzerinde sanal ortamı etkinleştirin:

```powershell
.\.venv\Scripts\Activate.ps1
```

Gerekli paketleri kurun:

```powershell
pip install -r requirements.txt
```

## Belgeleri Hazırlama

Kullanmak istediğiniz `.txt` belgelerini şu klasöre ekleyin:

```text
data/documents/
```

Belgelerde paragraflar arasında boş satır bulunmalıdır. Uygulama metinleri boş satırlara göre parçalara ayırır.

## Veritabanını Oluşturma

Belgeleri embedding vektörlerine dönüştürmek ve SQLite veritabanına kaydetmek için:

```powershell
python ingest_documents.py
```

Başarılı bir çalıştırma sonunda benzer bir çıktı görülür:

```text
3 belge bulundu.

foundry_local_notes.txt: 3 parça bulundu.
rag_notes.txt: 3 parça bulundu.
sqlite_notes.txt: 3 parça bulundu.

Belge işleme tamamlandı.
Toplam belge sayısı: 3
Toplam parça sayısı: 9
Veritabanı: data\rag_database.db
```

## Retrieval Sistemini Test Etme

Sadece belge arama sistemini çalıştırmak için:

```powershell
python retrieval.py
```

Örnek soru:

```text
RAG ne işe yarar?
```

Sistem en ilgili belge parçalarını, kaynak dosyalarını ve cosine similarity puanlarını gösterir.

## RAG Uygulamasını Çalıştırma

Ana uygulamayı başlatmak için:

```powershell
python rag_assistant.py
```

Program açıldığında terminal üzerinden soru sorabilirsiniz:

```text
Sorun: RAG nedir?
```

Örnek cevap:

```text
Retrieval-Augmented Generation, kısaca RAG, bir yapay zekâ modelinin cevap vermeden önce dış kaynaklardan bilgi bulmasını sağlayan bir yöntemdir.

Kullanılan kaynaklar:
1. rag_notes.txt
2. rag_notes.txt
```

Programdan çıkmak için:

```text
q
```

yazabilirsiniz.

## Benzerlik Eşiği

Uygulama, alakasız soruların yerel dil modeline gönderilmesini engellemek için minimum benzerlik eşiği kullanır.

```python
MIN_SIMILARITY = 0.45
```

En yüksek benzerlik puanı bu değerin altındaysa sistem şu cevabı verir:

```text
Bu sorunun cevabı mevcut belgelerde bulunmuyor.
```

Bu kontrol, modelin kendi genel bilgisini kullanarak belge dışında cevap üretmesini azaltır.

## Test Edilen Senaryolar

Başarılı cevaplanan sorular:

- RAG nedir?
- RAG ne işe yarar?
- RAG sisteminde kullanıcı sorusundan sonra ne yapılır?
- SQLite neden yerel uygulamalar için uygundur?
- SQLite ayrı bir sunucu gerektirir mi?
- Foundry Local ne işe yarar?
- Foundry Local internet olmadan çalışabilir mi?

Belgelerde olmadığı için reddedilen sorular:

- Python programlama dilini kim geliştirdi?
- Türkiye'nin başkenti neresidir?
- Yapay zekâ nedir?

## Mevcut Sınırlamalar

- Yalnızca `.txt` belgeleri desteklenmektedir.
- Metinler paragraf tabanlı olarak parçalanmaktadır.
- Arama işlemi tüm embedding vektörlerini belleğe alarak yapılmaktadır.
- Benzerlik eşiği sabit bir değerdir.
- Küçük belge koleksiyonları için uygundur.
- Kullanıcı arayüzü henüz terminal tabanlıdır.
- Sohbet geçmişi tutulmamaktadır.

## Gelecek Geliştirmeler

- Streamlit veya Gradio arayüzü eklemek
- PDF ve Word belgelerini desteklemek
- Chunk boyutunu ve overlap yapısını geliştirmek
- Sohbet geçmişi eklemek
- Kaynak metinleri arayüzde göstermek
- Otomatik test sistemi hazırlamak
- Daha büyük belge koleksiyonları için vektör veritabanı kullanmak
- Benzerlik eşiğini test sonuçlarına göre dinamik hâle getirmek
- Cevap doğruluğunu ölçen değerlendirme sistemi eklemek

## Tamamlanan Aşamalar

- Python proje ortamı oluşturuldu.
- Sanal ortam hazırlandı.
- Foundry Local SDK kuruldu.
- Yerel sohbet modeli çalıştırıldı.
- Embedding üretimi test edildi.
- Cosine similarity ile anlamsal arama yapıldı.
- SQLite veritabanı oluşturuldu.
- Belgeler parçalara ayrıldı.
- Toplam 9 belge parçası veritabanına kaydedildi.
- Retrieval sistemi oluşturuldu.
- Yerel LLM ile RAG akışı tamamlandı.
- Kaynak gösterimi eklendi.
- Alakasız sorular için benzerlik eşiği eklendi.
- Doğru ve yanlış soru senaryoları test edildi.
