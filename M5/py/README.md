# Projekt M5 mit Python

## Generelles

#### Zu optimierende Metrik
 * M5 Accuracy --> Weighted Root Mean Squared Scaled Error (RMSSE)
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
     
        
 ##### Ideen Max
 
 M.E. Dataset mit eventuell starker Saisonalität (Day of Week + Month) & Korrelation zwischen Produkten
 
 ---> ökonomisches Rational - komplementär Güter v.A. im Retail wie bspw. Grillwurst/Steaks & Grill etc. 
 
  * Dimensionality reduction? PCA or LDA 
  https://www.researchgate.net/publication/286485682_Dimensionality_Reduction_and_Filtering_on_Time_Series_Sensor_Streams
  
  * Regression Trees? LightGBM, XGBoost
  Stacked Models (ensembles) with scikit Pipeline? + Cross-Validation, but performance?
  
  * Neural Networks - LSTM CNN
  
  * Arima-Type Models- fast as MLH-estimator
  
  * Hybrid Models --> Refeed error into other model-types to estimate target variable additionally
       
    