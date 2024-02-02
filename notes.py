"""
unstack() yöntemi, Pandas veri çerçevelerinde ve Serilerinde çok seviyeli (hiyerarşik) indekslemeyi düzleştirmek için kullanılır. Genellikle çok seviyeli indekslemeli bir veri çerçevesini veya Serisini daha anlamlı veya kullanışlı bir şekilde analiz etmek veya görselleştirmek için düz bir yapıya dönüştürmek amacıyla kullanılır.

Özellikle çok seviyeli indekslemeye sahip bir veri çerçevesinde, unstack() kullanarak bir seviyeyi (level) sıradan sütunlara dönüştürebilirsiniz. Bu, veriyi daha kolay anlamak ve işlemek için kullanışlı olabilir.

İşte unstack() yönteminin kullanımı:

python
Copy code
import pandas as pd

# Örnek bir çok seviyeli indeksli veri çerçevesi
data = {
    'Ülke': ['Türkiye', 'Türkiye', 'ABD', 'ABD', 'Çin', 'Çin'],
    'Şehir': ['İstanbul', 'Ankara', 'New York', 'Los Angeles', 'Pekin', 'Şangay'],
    'Nüfus': [15000000, 5500000, 8400000, 3900000, 21500000, 24200000]
}

df = pd.DataFrame(data)
df = df.set_index(['Ülke', 'Şehir'])  # Çok seviyeli indeksleme

# Çok seviyeli indekslemeyi düzleştirme
df_unstacked = df.unstack()
print(df_unstacked)
Bu örnekte, unstack() yöntemi ile çok seviyeli indeksler düzleştirilir ve sonuç olarak bir düz yapıda bir veri çerçevesi elde edilir. Bu, daha sonra veriyi daha kolay analiz etmek veya görselleştirmek için kullanabilirsiniz.
"""

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################


"""
Sütun türleri arasında "Year" sütununun "int64" olduğunu göreceksiniz. Ancak, zaman serileriyle çalışmak için Pandas'ın farklı bir veri türü olan "datetime64" kullanmak istiyoruz.

Adım 5: "Year" sütununun veri türünü "datetime64" olarak dönüştürelim:

python
Copy code
crime['Year'] = pd.to_datetime(crime['Year'], format='%Y')
Bu kod, "Year" sütununun veri türünü "datetime64" olarak değiştirir. format='%Y' kullanarak, yıl bilgisi olduğunu belirtiyoruz. Dönüşüm tamamlandığında, "Year" sütunu artık bir tarih/saat sütunu olarak işlenecektir."""

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################


"""
# To learn more about .resample (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html)
# To learn more about Offset Aliases (http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)

# Uses resample to sum each decade
crimes = crime.resample('10Y').sum()

# Uses resample to get the max value only for the "Population" column
population = crime['Population'].resample('10AS').max()

# Updating the "Population" column
crimes['Population'] = population

crimes"""

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################


"""
dtype='l' ifadesi NumPy kütüphanesinde kullanılan bir parametredir ve verilerin hangi türde saklanacağını belirtir. dtype parametresi, veri türünü (data type) belirlemek için kullanılır. 'l' burada bir veri türünü temsil eder.

'l' (küçük "L") veri türü, "long integer" yani uzun tam sayı veri türünü temsil eder. Bu, büyük tam sayıları saklamak için kullanılır ve genellikle Python'daki int veri türünden daha büyük değerlerle çalışmak için kullanılır.

Örneğinizde np.random.randint işlevi ile rastgele uzun tam sayılar üretiliyor ve bu tam sayılar 'l' veri türünde saklanıyor. Bu, büyük aralıklardaki tam sayıları saklamak için kullanışlıdır."""

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

"""dropna pandas kütüphanesinde kullanılan bir işlevdir ve DataFrame veya Seri içerisindeki eksik (NaN) değerleri kaldırmak için kullanılır. Eksik değerler, veri analizi ve işlemler sırasında sorunlara neden olabilir, bu nedenle bu tür değerleri temizlemek önemlidir. dropna işlemini kullanarak aşağıdaki işlemleri yapabilirsiniz:

DataFrame'den Eksik Değerleri Kaldırma:


df.dropna()
Bu, DataFrame içindeki herhangi bir eksik değeri içeren satırları tamamen kaldırır.

Belirli Bir Sütunda Eksik Değerleri Kaldırma:

df.dropna(subset=['sütun_adı'])
Belirli bir sütundaki eksik değerleri kaldırmak için kullanılır.

Eksik Değerlerin Tüm Sütunlarda Kaldırılması:

df.dropna(axis=1)
Bu, tüm sütunlardaki eksik değerleri kaldırır.

Eksik Değerlerin Belirli Bir Eşik Değer Altında Kaldırılması:

df.dropna(thresh=2)
Bu, bir satırın en az belirli bir sayıda eksik olmadığı sürece satırları kaldırır.

Kopyalama Yapmadan Veriyi Değiştirme:
dropna varsayılan olarak veriyi değiştirmez, yani orijinal DataFrame üzerinde değişiklik yapmaz. Ancak, inplace=True parametresini kullanarak orijinal veriyi değiştirmesini sağlayabilirsiniz."""

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

""""year = monthly.resample('AS-JAN').sum()" ifadesi, aylık frekansta olan zaman serisi verisinin frekansını yıllık (year) olarak değiştirmek ve bu yıllık dönemler içindeki değerleri toplamak anlamına gelir. Ayrıca, yıl başlangıcını Ocak ayı olarak ayarlar.

Bu işlem, aylık verileri yıllık toplamlar halinde görmek veya yıllık analizler yapmak için kullanışlıdır. Özellikle zaman serisi verileriyle çalışırken, veriyi farklı frekansta yeniden örneklemek ve analiz etmek oldukça yaygındır.

Örnek bir Python kodu ile bu işlemi gerçekleştirebiliriz:

# Aylık veriyi yıllık toplamlara dönüştürelim (Ocak ayı başlangıcı)
yearly = monthly_time_series.resample('AS-JAN').sum()
"""

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

"""apple.resample('BM').mean()
ne demek
ChatGPT
Bu kod, bir zaman serisi verisi içindeki Apple hisse senedi fiyatlarının iş ayı (Business Month) bazında ortalamasını hesaplar.

'BM', iş ayını temsil eden bir resampling kodudur. İş ayı, hafta sonlarını ve tatil günlerini dikkate almadan sadece iş günlerini içeren bir ayı ifade eder. Örneğin, Ocak 2023 tarihli bir iş ayı, 1 Ocak'tan başlayarak tatil günleri olmadan iş günlerinin sonuna kadar olan süreyi kapsar.

.mean(), her iş ayı için hesaplanan Apple hisse senedi fiyatlarının ortalamasını alır.

Sonuç olarak, bu kod iş ayı bazında Apple hisse senedi fiyatlarının ortalama değerlerini hesaplar ve bu verileri bir zaman serisi olarak döndürür. Bu, hisse senedi fiyatlarının iş ayı içindeki eğilimlerini görmek veya analiz etmek için kullanışlı olabilir.
"""

#######################################################################################################################################
########################################################################################################################################
########################################################################################################################################

"""sep='|' ne demek
ChatGPT
"sep='|'" ifadesi, bir metin veya veri dosyasını okurken veya yazarken sütunları ayırmak için kullanılan ayırıcı karakterin ne olduğunu belirtir.
 Burada "|" karakteri sütunları ayırmak için kullanılan özel bir ayırıcı karakterdir. Bu karakter, verilerin düzgün bir şekilde sütunlara ayrıldığı 
 ve okunduğu veya yazıldığı bir dosya biçimini belirtir.

Örnek olarak, bir veri dosyası "|" karakteri ile ayrılmışsa ve bu verileri bir Python programında okumak istiyorsanız, sep='|' ifadesini kullanarak 
Python'a bu ayırıcı karakteri belirtirsiniz. Bu, verilerin doğru şekilde sütunlara ayrılmasına yardımcı olur."""

#######################################################################################################################################
########################################################################################################################################
########################################################################################################################################


rasgtgele numpy arrayi olusturma =
np.random.randint(1, 10, 10)
#vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

#access an element
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

#indexs must be within the range of the vector or they will produce and error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)

#vector slicing operations
a = np.arange(10)
print(f"a         = {a}")

#access 5 consecutive elements (start:stop:step)
c = a[2:7:1];     print("a[2:7:1] = ", c)

# access 3 elements separated by two
c = a[2:7:2];     print("a[2:7:2] = ", c)

# access all elements index 3 and above
c = a[3:];        print("a[3:]    = ", c)

# access all elements below index 3
c = a[:3];        print("a[:3]    = ", c)

# access all elements
c = a[:];         print("a[:]     = ", c)

a = np.array([1,2,3,4])
print(f"a             : {a}")
# negate elements of a
b = -a
print(f"b = -a        : {b}")

# sum all elements of a, returns a scalar
b = np.sum(a)
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2      : {b}")

#######################################################################################################################################
########################################################################################################################################
##########################################################################################################################

birden fazla csv okuyup tek pdf halıne donusturmek:
all_files = glob.glob(r'C:/Users/User/Desktop/Python' + "/*.csv")
all_df = pd.concat([pd.read_csv(f, delimiter=';') for f in all_files])
all_df

#######################################################################################################################################
########################################################################################################################################
##########################################################################################################################
sns.heatmap(df.isnull(), cbar=False) kodu eksık değerleri gorselleştirir

#######################################################################################################################################
########################################################################################################################################
#########################################################################################################################

kosullu değerlendirmeler içim loc kullanılır


MinMaxScaler('in fit ve transform metodları, 2 boyutlu bir array ya da DataFrame bekler.'
             ' Bu nedenle, tek bir sütunu seçerken, bu sütunu 2 boyutlu bir yapıda tutmak'
             ' için çift köşeli parantez kullanılır.)

Örnek olarak:

df["vote_count"] ifadesi bir pandas Series döndürür.
df[["vote_count"]] ifadesi ise bir pandas DataFrame döndürür.
fit ve transform metodlarına tek boyutlu bir Series verirseniz, hata alabilirsiniz.
Bu nedenle, bu metodları çağırırken çift köşeli parantez kullanarak sütunu DataFrame
formatında geçirmek önemlidir.

#######################################################################################################################################
########################################################################################################################################
########################################################################################################################################


Normallik incelemesi, birçok istatistiksel analizin temel varsayımıdır. Ancak, veri setinde aykırı değerlerin (outlier) varlığı, verinin normal dağılıma sahip olup olmadığının doğru bir şekilde değerlendirilmesini zorlaştırabilir. Bu nedenle normallik testi yapmadan önce aykırı değer incelemesi ve gerekiyorsa düzeltmesi yapılması önerilir.

1. Aykırı Değer Tespiti
Görselleştirme:
Boxplot (Kutu Grafiği): Aykırı değerleri tespit etmek için en yaygın yöntemlerden biridir. Boxplot'ta, kutunun dışında kalan değerler genellikle aykırı değer olarak kabul edilir.
Histogram ve Yoğunluk Grafiği: Değerlerin dağılımını görselleştirmek için kullanılır. Eğer grafikte belirgin "kuyruklar" veya "tepecikler" varsa bu, aykırı değerlerin varlığına işaret edebilir.
İstatistiksel Yöntemler:
Z-Skor: Bir veri noktasının, veri setinin ortalamasına göre standart sapma cinsinden ne kadar uzakta olduğunu ölçer. Örneğin, z-skoru 2.5'tan büyük olan değerler genellikle aykırı değer olarak kabul edilir.
IQR (Çeyrekler Açıklığı): Verinin alt %25'lik (Q1) ve üst %25'lik (Q3) çeyrekleri arasındaki farktır. Genellikle [Q1 - 1.5IQR, Q3 + 1.5IQR] dışında kalan değerler aykırı değer olarak kabul edilir.

######################################################################################################################################
########################################################################################################################################
###########################################################################################################



corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

"""Bu kod parçasını adım adım açıklayalım:

final_df.T: final_df DataFrame'ini transpoze eder, yani sütunlarla satırları yer değiştirir.

.corr(): Transpoze edilmiş DataFrame üzerinde korelasyon hesaplaması yapar. Bu, sütunlar arasındaki (orijinal DataFrame'de satırlar arasındaki) korelasyon değerlerini verir.

.unstack(): Hiyerarşik indeksi (multi-index) olan bir DataFrame'ı tek seviyeli bir indekse dönüştürür. Bu adımda, korelasyon değerleri için çiftler halinde sütun ve satır indeksleri olan bir Series elde edilir.

.sort_values(): Elde edilen Series'teki değerleri sıralar.

.drop_duplicates(): Sıralanmış Series'teki tekrar eden değerleri kaldırır. Bu adım, korelasyon matrisinin üst ve alt üçgeni arasındaki tekrar eden korelasyon değerlerini kaldırmak için gereklidir, çünkü bir korelasyon matrisi simetriktir.

Sonuç olarak, corr_df adlı değişken, final_df DataFrame'indeki satırlar arasındaki benzersiz korelasyon değerlerini sıralı bir şekilde içerir. Bu, hangi satırların birbiriyle ne kadar benzer veya farklı olduğunu görmek için kullanılabilir."""

#####################################################################################################################################
########################################################################################################################################
###########################################################################################################


Eğer "confidence" yüksek ancak "lift" düşükse, bu durum şu anlama gelir:
Bir öğenin alınmasının diğer öğenin de alınmasına olan yüksek olasılığına("confidence")
rağmen, bu iki öğe birbiriyle gerçekte ne kadar bağımlı olduğunu ifade eden "lift"
değeri düşüktür. Bu, bu iki öğenin birlikte alınma olasılığının, rastgele birlikte
alınma olasılığına göre daha az olduğu anlamına gelir. Yani, bu iki öğenin birlikte
alınması özel bir ilişki göstermez, çünkü birlikte alınma olasılıkları beklenenden düşüktür.

#####################################################################################################################################
#######################################################################################################################

recommendation_list = [list(rule.consequents)[0]
    for rule in sorted_rules.itertuples()
    if product_id in rule.antecedents]


recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

####################################################################################################################################
#######################################################################################################################


 df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
"""Kod parçası df.Name.str.extract(' ([A-Za-z]+)\.', expand=False), bir pandas DataFrame'inin "Name"
 sütunundaki veriler üzerinde çalışır ve her isimdeki bir kelimeyi belirli bir desene göre çıkarmak 
 için kullanılır. Buradaki desen, bir regex (regular expression) desenidir ve parantez içindeki kısımları açıklayayım:

([A-Za-z]+): Bu kısım, en az bir harf içeren bir kelimeyi yakalar. [A-Za-z] kısmı, büyük veya küçük harf A'dan Z'ye
 kadar olan herhangi bir harfi ifade eder. + işareti ise bir veya daha fazla karakterin varlığını belirtir. Yani bu kısım,
  bir veya daha fazla harften oluşan kelimeleri yakalamak için kullanılır.

\.: Regex'de nokta karakteri (.) özel bir anlama sahiptir ve genellikle herhangi bir karakteri ifade eder.
 Ancak burada bir backslash (\) ile öncesinde, yani \. olarak kullanılmıştır, bu sayede regex deseninde gerçek
  bir nokta karakteri olarak değerlendirilir.

expand=False: str.extract fonksiyonunun expand parametresi, eğer False olarak ayarlanırsa, sonucu tek bir sütun
 olarak döndürür. Eğer True olsaydı (varsayılan), her bir yakalanan grup (parantez içindeki desenler) ayrı bir sütuna karşılık gelirdi.

Özetle, bu regex deseni bir DataFrame sütunundaki metinler içerisinde, nokta (.) ile biten ve sadece harflerden 
oluşan kelimeleri çıkarmak için kullanılır. Tipik bir kullanımı, insan isimlerinden unvanları (Mr., Mrs., Dr. vb.) çıkarmaktır.
 Örneğin, "Mr. Smith" metni içinde "Mr" kelimesini yakalayacaktır."""


####################################################################################################################################
#######################################################################################################################


# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

####################################################################################################################################
################################ Linear Regression Assumption #######################################################################
Hata terimlerinin normal dağılıma ve homoscedasticity varsayımına uygunluğunu kontrol etmek için bazı yöntemler ve grafikler kullanabiliriz.
İşte bu kontrolleri Python'da gerçekleştirmek için kullanılabilecek bazı yöntemler:

Normality (Normallik) Kontrolü:
QQ Plot:
Hata terimlerinin normal dağılıma ne kadar uygun olduğunu görsel olarak değerlendirebiliriz.
statsmodels kütüphanesinin qqplot fonksiyonu kullanılabilir.

import statsmodels.api as sm
import matplotlib.pyplot as plt

residuals = model.resid
fig, ax = plt.subplots(figsize=(8, 4))
sm.qqplot(residuals, line='s', ax=ax)
plt.show()
####################################################################################################################################

Shapiro-Wilk Testi:
Hata terimlerinin normal dağılıma uygunluğunu test etmek için Shapiro-Wilk testi kullanılabilir.
scipy kütüphanesinde bu test bulunmaktadır.

from scipy.stats import shapiro

stat, p_value = shapiro(residuals)
print(f"Shapiro-Wilk Test Statistic: {stat}, p-value: {p_value}")
####################################################################################################################################

Homoscedasticity (Homojenlik) Kontrolü:
Residuals vs Fitted Plot:
Hata terimlerinin bağımsız değişkenlere göre yayılımını kontrol eden bir grafiktir.
Hata terimlerinin düzgün bir şekilde dağılıp dağılmadığını görmek için kullanılabilir.

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(model.fittedvalues, residuals)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()
####################################################################################################################################

Goldfeld-Quandt Testi:
Heteroscedasticity'i test etmek için kullanılabilir.
statsmodels kütüphanesinde bu test bulunmaktadır.

from statsmodels.stats.diagnostic import het_goldfeldquandt

test_stat, p_value, _ = het_goldfeldquandt(residuals, model.model.exog)
print(f"Goldfeld-Quandt Test Statistic: {test_stat}, p-value: {p_value}")
####################################################################################################################################

eğer hata terimleri normal dağılıma veya homoscedasticity varsayımlarına uymuyorsa, aşağıdaki alternatif yöntemlere başvurabilir veya modelinizi düzeltebilirsiniz:

Heteroscedasticity ile Başa Çıkma:
Weighted Least Squares (WLS):
Heteroscedasticity durumunda, gözlemlere farklı ağırlıklar atanarak regresyon modeli tahmin edilebilir. Bu, heteroscedasticity'nin şiddetini azaltabilir.
statsmodels kütüphanesinde WLS kullanabilirsiniz.

from statsmodels.regression.robust_linear_model import WLS

weights = 1 / residuals.var()  # Ağırlıkları uygun bir şekilde hesaplayın
wls_model = WLS(y, X, weights=weights).fit()
####################################################################################################################################

Transformations:
Bağımlı değişken veya bağımsız değişkenler üzerinde dönüşümler yaparak heteroscedasticity'i düzeltebilirsiniz. Örneğin, bağımlı değişkeni logaritma alabilirsiniz.

y_transformed = np.log(y)
model_transformed = sm.OLS(y_transformed, X).fit()
####################################################################################################################################

Normallik ile Başa Çıkma:
Transformations:
Bağımlı değişken üzerinde dönüşümler yaparak hata terimlerinin normal dağılıma daha yakın olmasını sağlayabilirsiniz.

y_transformed = np.sqrt(y)  # Veya başka bir dönüşüm
model_transformed = sm.OLS(y_transformed, X).fit()
####################################################################################################################################

Robust Regression:
Heteroscedasticity veya aykırı değerlere karşı daha dirençli olan robust regresyon yöntemlerini kullanabilirsiniz.

robust_model = sm.RLM(y, X).fit()

Bu yöntemlerden birini veya birkaçını deneyerek modelinizin performansını iyileştirebilirsiniz. Ancak, dikkatli olunmalı ve her türlü dönüşüm veya düzeltici yöntemin etkilerini değerlendirmek için model performansını tekrar değerlendirmeniz gerekmektedir.
####################################################################################################################################
####################################################################################################################################

Pythonda aşırı öğrenmeyi kontrol etmek için bahsettiğim yöntemleri uygulayabilirsiniz. İşte bazı örnekler:

Eğitim ve Test Hatasının Karşılaştırılması:

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Veriyi eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelinizi eğitin
model.fit(X_train, y_train)

# Eğitim ve test setlerinde tahmin yapın
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Hataları hesaplayın
train_error = mean_squared_error(y_train, y_train_pred)
test_error = mean_squared_error(y_test, y_test_pred)

print(f"Eğitim Hatası: {train_error}")
print(f"Test Hatası: {test_error}")
####################################################################################################################################

Validation Set Kullanımı:
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Veriyi eğitim, test ve validation setlerine bölelim
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Modelinizi eğitin
model.fit(X_train, y_train)

# Validation setinde performansı değerlendirin
y_val_pred = model.predict(X_val)
validation_error = mean_squared_error(y_val, y_val_pred)

print(f"Validation Set Hatası: {validation_error}")
####################################################################################################################################

Öğrenme Eğrisi Analizi:
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Öğrenme eğrisini çizin
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Hataları ortalamak için kullanabilirsiniz
train_errors_mean = -train_scores.mean(axis=1)
test_errors_mean = -test_scores.mean(axis=1)

# Eğitim ve test hatalarını çizin
plt.plot(train_sizes, train_errors_mean, label='Eğitim Hatası')
plt.plot(train_sizes, test_errors_mean, label='Test Hatası')
plt.legend()
plt.show()
####################################################################################################################################

Çapraz Doğrulama (Cross-Validation)
from sklearn.model_selection import cross_val_score

# Çapraz doğrulama ile modelin performansını değerlendirin
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Hataları ortalamak için kullanabilirsiniz
cv_errors_mean = -cv_scores.mean()

print(f"Çapraz Doğrulama Hatası: {cv_errors_mean}")