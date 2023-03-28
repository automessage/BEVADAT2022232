# load_csv
Betölti a paraméterként kapott útvonalról a csv file-t, DataFrame-mé alaíktja, majd megkeveri a sorokat, hogy ha a célváltozó alapján sorrendbe lenne rendezve, akkor a teszt és ellenőrző adatok szétosztásánál diverzek legyenek az adataink.

# train_test_split
Felosztja a DataFrame-et 2 x 2 felé:
1. Tanító adatok
    - bemeneti paraméterek
    - kimeneti paraméter
2. Ellenőrző adatok
    - bemeneti paraméterek
    - kimeneti paraméter

Erre azért van szükség, hogy a tanítás után tesztelhető legyen ismert kimeneteken a tanítás sikeressége/pontossága.

# euclidean
Euklidészi algoritmust használva kiszámítja egy vizsgálandó sor bemeneti paramétereihez vett távolságot a tanító adatok minden sorának bemeneti paramétereihez. Ezt úgy teszi meg, hogy az egyes attribútumok különbségét veszi, majd ezek négyzetét összegzi, mely összegből négyzetgyököt von.

# predict
A tesztelendő sorokra egyenként meghívja az `euclidean` metódust, ami visszaadja a távolságokat egy vektorban, melyeket rendez (a tanító adatok kimeneteivel összerendelve) növekvő sorrendbe, majd a `k` paraméter alapján a legelső `k` darab sorban leggyakoribban előforduló kimetelt kiválasztja a vizsgált sorhoz.

# accuracy
A `predict` metódus által megtippelt kimeneteleket összehasonlítjuk a a teszt adatok valós kimeteivel és visszadjuk, hogy hány %-ban sikerült eltalálni a kimenetelt.

# confusion_matrix
Visszaad egy mátrixot, mely megmutatja, hogy a valós és valótlan tippek hogyan oszlanak szét a klasszifikációs csoportok között

# best_accuracy
Visszad egy adatpárost, melyben az első a `k` paramétere, a második pedig a pontossága a legpontossab predikációnak.