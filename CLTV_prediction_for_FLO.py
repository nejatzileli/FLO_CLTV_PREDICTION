import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', None)

# 1. Gorev : Titanic veri setini yukleyiniz. Seaborn kutuphanesinden, sik kullanilanlar bolumunden Titanic verisetini df icine kaydettim.

df = sns.load_dataset("titanic")

# 2. Gorev : Titanic verisetindeki erkek ve kadin yolcu sayilarini bulunuz. Once male/female bulunan "sex" sutununu sectim. Daha sonra PandasObject class'inin icindeki "value_counts" metodunu kullanarak
# verisetindeki essiz (unique) degerleri ve sayilarini buldum.

df['sex'].value_counts()

# 3. Gorev: Her bir sutuna ait unique değerlerin sayısını bulunuz. "nunique" methodu = "number of unique". Her bir sutuna ait unique degerler bulundu.

df.nunique()

# 4. Gorev: pclass değişkeninin unique değerlerinin sayısını bulunuz.

df['pclass'].nunique()

# 5. Gorev: pclassveparch değişkenlerinin unique değerlerinin sayısını bulunuz. Iki degiskenin cevresine '(, )' arasina 'and' ifadesi yerlestirdim.
# Boylece iki degiskeninde toplam unique degerlerini verdi.

             #df[('pclass') and ('parch')].nunique() # bu yanls
df[['pclass','parch']] #birden fazla degisken girersek liste yapamadigi icin dataframe ciktisi almasi lazim


# 6. Gorev:  embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz

df['embarked'].dtypes # Dataframe'den embarked degiskenini sectim ve tipini dtype methodu ile kontrol ettim. Tipi Object cikti.

df['embarked']= df['embarked'].astype('category') # astype() methodu kullanarak degistenin tipini object'ten category'ye cevirdim.
df.info() # burada da goruelecegi uzere, embarked degiskeni 'category' tipine donustu

# 7. Gorev: embarked değeri C olanların tüm bilgelerini gösteriniz.

df[df['embarked'] == 'C']

# 8. Gorev : embarked değeri S olmayanların tümbilgelerinigösteriniz.

df[df['embarked'] != 'S']
df[df['embarked'] != 'S']['embarked'].unique()
# 9. Gorev : Yaşı30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.

df[(df['sex'] == 'female') & (df['age'] <30.0)]

# 10. Gorev : Fare'i 500'den büyük veya yaşı 70’den büyük yolcuların bilgilerini gösteriniz.

df[(df['fare'] >500) | (df['age'] > 70)]

# 11. Gorev : Her bir değişkendeki boş değerlerin toplamını bulunuz.

df.isnull().sum()

# 12. Gorev : who değişkeninidataframe’dençıkarınız.

df.drop('who', axis=1,inplace = True)

df.columns

#13. Gorev : deck değikenindekiboşdeğerlerideck değişkeninençoktekraredendeğeri(mode) iledoldurunuz

df['deck'].fillna(df['deck'].mode()[0], inplace= True)

#14. Gorev :  age değikenindeki boşdeğerleri age değişkenin medyanı ile doldurunuz.

df['age'].fillna(df['age'].median(), inplace = True)

#15 Gorev : survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerinibulunuz

df.groupby(["pclass", "sex"]).agg({'survived': ['sum', 'count', 'mean']})

#16 Gorev : 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 verecek bir fonksiyon yazın.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz.(apply ve lambda yapılarını kullanınız)

df['age_flag'] = df['age'].apply(lambda x : 1 if(x<30) else 0)

df

#17 Gorev :  Seaborn kütüphanesi içerisinden Tipsveri setini tanımlayınız

dff = sns.load_dataset('tips')

#18 Gorev : Time değişkeninin kategorilerine(Dinner, Lunch) göre total_billdeğeri nin sum, min, max ve mean değerlerini bulunuz.

dff.groupby('time').agg({'total_bill' : ['sum','min','max', 'mean']})

#19 Gorev : Day vetime’a göre total_billdeğerlerinin sum, min, max ve mean değerlerini bulunuz.

dff.groupby(['day','time']).agg({'total_bill' : ['sum','min','max', 'mean']})

#20 Gorev:  Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre sum, min, max ve mean değerlerini bulunuz.

a = dff[(dff['sex'] == 'Female') & (dff['time'] == 'Lunch')].groupby('day').agg({'total_bill' : ['sum','min', 'max', 'mean'],
                                                                                 'tip': ['sum','min', 'max', 'mean']})
a
#21 Gorev: size'i3'ten küçük, total_bill'i10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)

dff.loc[(dff['size']<3) & (dff['total_bill']>10),'total_bill'].mean()

#22 Gorev:total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği total bill ve tip'in toplamını versin.

dff['total_bill_tip_sum'] = dff['total_bill']+ dff['tip']

#23 Gorev :  Total_bill değişkeninin kadın ve erkek için ayrı ayrı ortalamasını bulunuz. Bulduğunuz ortalamaların altında olanlara 0,
# üstünde ve eşit olanlara 1 verildiği yeni bir total_bill_flag
# değişkeni oluşturunuz

def means(sex, total_bill):
    if sex == 'male':
        mean_m = dff[dff['sex'] == 'Male']['total_bill'].mean()
        if total_bill < mean_m:
            return 0
        else:
            return 1
    else:
        mean_f = dff[dff['sex'] == 'Female']['total_bill'].mean()
        if total_bill < mean_f:
            return 0
        else:
            return 1


dff['total_bill_flag']=dff.apply(lambda x: means(x['sex'],x['total_bill']),axis=1)

dff

dff.columns

#24 Gorev: total_bill_flag değişkenini kullanarak cinsiyetlere göre ortalamanın altında ve üstünde olanların sayısını gözlemleyiniz.
dff.groupby(['sex', 'total_bill_flag']).agg({'sex' : 'count'})


#25 Gorev : Total_bill_sum degiskenine gore veriyi sirala, ilk 30 u al.

new_dff = dff.sort_values('total_bill_tip_sum', ascending= False)[:30]

new_dff

