# Projekt M5 mit Python

## Generelles

#### Zu optimierende Metrik
 * M5 Accuracy --> Weighted Root Mean Squared Scaled Error (RMSSE) 
            --> custom loss function in sklearn
 * M5 Uncertainty --> Weighted Scaled Pinball Loss (WSPL)
 
#### Gegenstand der Challenge + Datensatzes

 * Walmart Sales Datensatz für mehrere Produktypen + Stores
 * Stores befinden sich in 3 US-Staaten (Calfornia, Texas, Wisconsin)
 * Erweitert durch Variablen wie "day of the week", "Promotions" & "special events"
 
 #### Ziel der Challenge
 * Vorhersage von Produktumsätzen je Store über eine Periode von 28 Tagen

 ##### Dateien:
    sample_submisssion.csv
        Spalten = Vorhersage für 28 Tage (F1 - F28)
        Zeilen = Composite Key - Produktkategorie + Produkt_ID + State_ID + DataSetType
 
    sales_train_validation:
        Spalten = Tage 
        Zeilen = Produkte_Stores
        
    calendar.csv
        Mapping von d_1 - d_1969 mit eigentlichem Datum
        Events per day
        
     sell_prices.csv
        Preis pro Produkt pro Storelocation pro Jahr_Woche
     
 ##### Vorbereitung Datensatz & 
 * Zusammenführung der einzelnen Dateien zu einem "Masterdatensatz"
    --> Produktverkäufe multiplizieren mit den Verkaufspreisen zum Zeitpunkt 't' pro Storelocation
    
  * Event-Dummy Variablen + Dates DS hinzufügen 
  
  * Da Verkaufsdaten zu "noisy", wahrscheinlich hilfreich rollierende Durschnitte zu bilden


 ##### Ideen Max
 
 M.E. Dataset mit eventuell starker Saisonalität (Day of Week + Month) & Korrelation zwischen Produkten
 
 ---> ökonomisches Rational - komplementär Güter v.A. im Retail wie bspw. Grillwurst/Steaks, Grill & Camping Stühle etc. 
 
  * Dimensionality reduction? PCA or LDA 
  https://www.researchgate.net/publication/286485682_Dimensionality_Reduction_and_Filtering_on_Time_Series_Sensor_Streams
  
  * Regression Trees with Gradient Boosting? LightGBM, XGBoost
  Stacked Models (ensembles) with sklearn Pipeline? + Cross-Validation, but performance?
  
  * Neural Networks - LSTM CNN
  
  * Arima-Type Models- fast as MLH-estimator
  
  * Hybrid Models --> Refeed error into other model-types to estimate target variable additionally
       
    