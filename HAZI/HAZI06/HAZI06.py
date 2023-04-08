"""
1. Értelmezd az adatokat!!!

2. Írj egy osztályt a következő feladatokra:  
     - Neve legyen NJCleaner és mentsd el a NJCleaner.py-ba. Ebben a fájlban csak ez az osztály legyen.
     - Konsturktorban kapja meg a csv elérési útvonalát és olvassa be pandas segítségével és mentsük el a data (self.data) osztályszintű változóba 
     - Írj egy függvényt ami sorbarendezi a dataframe-et 'scheduled_time' szerint növekvőbe és visszatér a sorbarendezett df-el, a függvény neve legyen 'order_by_scheduled_time' és térjen vissza a df-el  
     - Dobjuk el a from és a to oszlopokat, illetve a nan-okat és adjuk vissza a df-et. A függvény neve legyen 'drop_columns_and_nan' és térjen vissza a df-el  
     - A date-et alakítsd át napokra, pl.: 2018-03-01 --> Thursday, ennek az oszlopnak legyen neve a 'day'. Ezután dobd el a 'date' oszlopot és térjen vissza a df-el. A függvény neve legyen 'convert_date_to_day' és térjen vissza a df-el   
     - Hozz létre egy új oszlopot 'part_of_the_day' névvel. A 'scheduled_time' oszlopból számítsd ki az alábbi értékeit. A 'scheduled_time'-ot dobd el. A függvény neve legyen 'convert_scheduled_time_to_part_of_the_day' és térjen vissza a df-el  
         4:00-7:59 -- early_morning  
         8:00-11:59 -- morning  
         12:00-15:59 -- afternoon  
         16:00-19:59 -- evening  
         20:00-23:59 -- night  
         0:00-3:59 -- late_night  
    - A késéeket jelöld az alábbiak szerint. Az új osztlop neve legyen 'delay'. A függvény neve legyen pedig 'convert_delay' és térjen vissza a df-el
         0 <= x 5  --> 0  
         5 <= x    --> 1  
    - Dobd el a felesleges oszlopokat 'train_id' 'scheduled_time' 'actual_time' 'delay_minutes'. A függvény neve legyen 'drop_unnecessary_columns' és térjen vissza a df-el
    - Írj egy olyan metódust, ami elmenti a dataframe első 60 000 sorát. A függvénynek egy string paramétere legyen, az pedig az, hogy hova mentse el a csv-t (pl.: 'data/NJ.csv'). A függvény neve legyen 'save_first_60k'. 
    - Írj egy függvényt ami a fenti függvényeket összefogja és megvalósítja (sorbarendezés --> drop_columns_and_nan --> ... --> save_first_60k), a függvény neve legyen 'prep_df'. Egy paramnétert várjon, az pedig a csv-nek a mentési útvonala legyen. Ha default value-ja legyen 'data/NJ.csv'

3.  A feladatot a HAZI06.py-ban old meg.
    Az órán megírt DecisionTreeClassifier-t fit-eld fel az első feladatban lementett csv-re. 
    A feladat célja az, hogy határozzuk meg azt, hogy a vonatok késnek-e vagy sem. 0p <= x < 5p --> nem késik, ha 5 < x --> késik.
    Az adatoknak a 20% legyen test és a splitelés random_state-je pedig 41 (mint órán)
    A testset-en 80% kell elérni. Ha megvan a minimum százalék, akkor azzal paraméterezd fel a decisiontree-t és azt kell leadni.

    A leadásnál csak egy fit kell, ezt azzal a paraméterre paraméterezd fel, amivel a legjobb accuracy-t elérted.

    A helyes paraméter megtalálásához használhatsz grid_search-öt.
    https://www.w3schools.com/python/python_ml_grid_search.asp 

4.  A tanításodat foglald össze 4-5 mondatban a HAZI06.py-ban a fájl legalján kommentben. Írd le a nehézségeket, mivel próbálkoztál, mi vált be és mi nem. Ezen kívül írd le 10 fitelésed eredményét is, hogy milyen paraméterekkel probáltad és milyen accuracy-t értél el. 
Ha ezt feladatot hiányzik, akkor nem fogadjuk el a házit!

HAZI06-
    -NJCleaner.py
    -HAZI06.py

##################################################################
##                                                              ##
## A feladatok közül csak a NJCleaner javítom unit test-el      ##
## A decision tree-t majd manuálisan fogom lefuttatni           ##
## NJCleaner - 10p, Tanítás - acc-nál 10%-ként egy pont         ##
## Ha a 4. feladat hiányzik, akkor nem tudjuk elfogadni a házit ##
##                                                              ##
##################################################################
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.DecisionTreeClassifier import DecisionTreeClassifier
from src.Loader import Loader
from src.GridSearch import DecesionTreeGridSearch

loading = Loader('Reading CSV')
loading.start()
data = pd.read_csv("data/NJ.csv")
loading.terminate()
#print(data)

loading = Loader('Splitting data')
loading.start()
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.2, random_state=41)
loading.terminate()

loading = Loader('Fitting tree')
loading.start()
classifier = DecisionTreeClassifier(min_samples_split=13, max_depth=10)
classifier.fit(X_train, Y_train)
loading.terminate()

loading = Loader('Predicting on test data')
loading.start()
Y_pred = classifier.predict(X_test)
loading.terminate()

print('Accuracy: ' + str(accuracy_score(Y_test, Y_pred)))

"""
Mégységre és a szétosztási értékre IS figyelni kell

Mélység egyenesen arányos a túltanításal
Szétosztási érték fordítottan arányos a túltanítással

A kettő megfelelő egyensúlyát kell megtalálni.
"""

# gs = DecesionTreeGridSearch(possible_split_values=[5], possible_depth_values=range(7, 12))
# gs = DecesionTreeGridSearch(possible_split_values=range(6, 11), possible_depth_values=[10])
# gs = DecesionTreeGridSearch(possible_split_values=range(11, 15), possible_depth_values=[10])
# gs = DecesionTreeGridSearch(possible_split_values=range(15, 20), possible_depth_values=[10])

# best_result, results = gs.searchBestDepth(X_train, X_test, Y_train, Y_test)

# print('------------------------------------')
# print(best_result)
# print('------------------------------------')
# print(results)

"""
Első körben a fit-elésnél a mélységet nagyobb nagyságrendi léptékkel próbáltam meghatározni, hogy a pontossági görbe hol fordul vissza.
Itt az egyes fit-ek több időt vettek igénybe, így írtam egy kisebb loading jelzőt, amit egy külön szálon elindítottam, hogy képben legyek a futások hosszával.
Ennél a pontnál észrevettem, hogy 2-3 percig futnak a fit-ek, ami önmagában nem olyan hosszú idő, de több paraméterezés teszteléséhez ez igencsak összeadódik.
A javasolt grid search metodóligát átnézve írtam egy osztályt, ami a tesztelést elvégzi a beadott paraméterek listáján.
Itt észrevettem, hogy hibát dob a 'DecesionTreeClassifier' osztály egy bizonyos mélység felett.
A hiba okát feltártam és módosítottam az osztályon, ez után kezdődhetett a 7-nél magasabb mélységekre való fit-elések tesztelése, mivell itt még látszott, hogy a pontosság nő.
A tesztek alapján a 9-es mélységnél fordult vissza a pontossági görbe a teszt adatokon. Ezután a split értékét pontosítottam, melyre a 9 és 10-es érték ugyanazt az eredményt hozta, utána csökkenni kezdett a pontosság.
A feladat leírásában 80%-os pontosság van megadva, de csak 79,30172485626198% volt a legmagasabb amit el tudtam érni a paraméterezéssel.
Itt vettem észre, hogy az adatoknál a fejlécet a csv beolvasásánál a header=None paraméter miatt adatsorként olvasta be, ami elvitte az egész folyamatot.
Így újra megpróbálom meghatározni a pontossági görbe forduló pontját. 5 - 15 mélységig a grid search segítségével ellenőrzöm a pontosságokat.
A tesztek alapján a 10-es mélységre lett a legmagasabb eredmény. Erre a mélységre próbáljuk meghatározni a megfelelő vágási értéket, hogy a pontosságot maximalizáljuk.
Első körben 6-tól 11-ig próbálom meghatározni, a görbe irányváltásának pontját a 10-es mélységen.
Itt 7-es és 10-es vágás egyforma eredményt hozott, így ellenőrzöm a 11-15 intervallumot, hogy milyen irányba halad.
80 % a legjobb elért eredmény a 13 vágási és 10-es mélységgel
15-20-ig ellenőrzöm, hogy a görbe hogyan halad tovább, hátha van pontosabb eredmény.
Itt már csökkenő tendenciát mutat, így a legjobb talált eredményem:

##################################
###### min_samples_split=13 ######
###### max_depth=10         ######
###### accuracy: 80%        ######
##################################

A test_results.txt tartalmazza a próbálkozásaim eredményeit, ezek közül néhány:
    | split | depth | accuracy |
    |   5   |   8   | 79,55%   |
    |   5   |   9   | 79,775%  |
    |   5   |   10  | 79,9417% |
    |   5   |   11  | 79,8417% |
    |   5   |   12  | 79,4417% |
    
    |   9   |   10  | 79,975%  |
    |   10  |   10  | 79,9833% |
    |   11  |   10  | 79,9917% |
    |   14  |   10  | 79,9833% |
    |   15  |   10  | 79,9917% |
    |   17  |   10  | 79,9833% |
"""