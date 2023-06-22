#
#       Basic functions to read a csv file, label encoding
#       and column filtering.
#
#       computation of the evaluation metrics (R, P, F1 score, FPR)
#
#       Software is distributed without any warranty;
#       bugs should be reported to antonio.pecchia@unisannio.it
#

import pandas as pd
import numpy as np

from sklearn.preprocessing import MaxAbsScaler, StandardScaler, Normalizer, RobustScaler, MinMaxScaler, Binarizer

# 90 columns ('Label' included)

names =[ 'id','Flow ID','Src IP','Src Port','Dst IP','Dst Port','Protocol','Timestamp','Flow Duration',
         'Total Fwd Packet','Total Bwd packets','Total Length of Fwd Packet','Total Length of Bwd Packet',
         'Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std',
         'Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std',
         'Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min',
         'Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean',
         'Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags',
         'Fwd RST Flags','Bwd RST Flags','Fwd Header Length','Bwd Header Length','Fwd Packets/s','Bwd Packets/s',
         'Packet Length Min','Packet Length Max','Packet Length Mean','Packet Length Std','Packet Length Variance',
         'FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count',
         'CWR Flag Count','ECE Flag Count','Down/Up Ratio','Average Packet Size','Fwd Segment Size Avg',
         'Bwd Segment Size Avg','Fwd Bytes/Bulk Avg','Fwd Packet/Bulk Avg','Fwd Bulk Rate Avg','Bwd Bytes/Bulk Avg',
         'Bwd Packet/Bulk Avg','Bwd Bulk Rate Avg','Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets',
         'Subflow Bwd Bytes','FWD Init Win Bytes','Bwd Init Win Bytes','Fwd Act Data Pkts','Fwd Seg Size Min',
         'Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min',
         'ICMP Code','ICMP Type','Total TCP Flow Time','Label']


#  81 features being used (NOTE: 'Label' is excluded too)
# ( x1,x2,...,xn)

features =[ 'Flow Duration',
         'Total Fwd Packet','Total Bwd packets','Total Length of Fwd Packet','Total Length of Bwd Packet',
         'Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std',
         'Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std',
         'Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min',
         'Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean',
         'Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags',
         'Fwd RST Flags','Bwd RST Flags','Fwd Header Length','Bwd Header Length','Fwd Packets/s','Bwd Packets/s',
         'Packet Length Min','Packet Length Max','Packet Length Mean','Packet Length Std','Packet Length Variance',
         'FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count',
         'CWR Flag Count','ECE Flag Count','Down/Up Ratio','Average Packet Size','Fwd Segment Size Avg',
         'Bwd Segment Size Avg','Fwd Bytes/Bulk Avg','Fwd Packet/Bulk Avg','Fwd Bulk Rate Avg','Bwd Bytes/Bulk Avg',
         'Bwd Packet/Bulk Avg','Bwd Bulk Rate Avg','Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets',
         'Subflow Bwd Bytes','FWD Init Win Bytes','Bwd Init Win Bytes','Fwd Act Data Pkts','Fwd Seg Size Min',
         'Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min',
         'ICMP Code','ICMP Type','Total TCP Flow Time']

def transformData ( training, validation, test ):

    # csv -> dataframe
    dfTrain = pd.read_csv ( training, names=names, header=None, sep=',', index_col=False, dtype='unicode')
    dfValidation = pd.read_csv ( validation, names=names, header=None, sep=',', index_col=False, dtype='unicode')
    dfTest = pd.read_csv ( test, names=names, header=None, sep=',', index_col=False, dtype='unicode')

    x_train, y_train, l_train = getXY(dfTrain)
    x_val, y_val, l_val = getXY(dfValidation)
    x_test, y_test, l_test = getXY(dfTest)

    # Facciamo lo scaling dei dati (stesso problema del biplot)
    # Normalizzato al valore massimo
    scaler = MaxAbsScaler ( )

    # Fit determina il massimo per ogni colonna e trasform scala
    x_train = scaler.fit_transform(x_train)

    # Dobbiamo scalare nella stessa sessione anche gli altri
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return x_train, y_train, l_train, x_val, y_val, l_val, x_test, y_test, l_test

def getXY ( inDataframe ):

    # Abbiamo preso di tutto il CSV le colonne di interesse e le abbiamo messe nella x
    # 1.) inDataframe -> (solo le features)
    x = inDataframe[ features ].values.astype( float)

    # 2.) Y  ( 1, 0) -> BENIGN   / (0,1) -> DoS
    # Isoliamo la colonna label
    bitA = np.where ( inDataframe['Label'] == 'BENIGN', 1, 0)
    bitB = np.where ( inDataframe['Label'] == "BENIGN", 0, 1)

    y = np.column_stack(( bitA, bitB ))

    # 3.) Label originali forma stringa
    l = inDataframe['Label'].values

    return x, y, l

#-----------------------------------------------#
#           instructor-provided code            #
#-----------------------------------------------#
def evaluatePerformance(outcome, evaluationLabels):

        # outcome: boolean      TRUE    ->  BENIGN
        #                       FALSE   ->  ATTACK

        # evaluationLabels:     original labels


        eval = pd.DataFrame( data={'prediction':outcome, 'Class':evaluationLabels} )

        TN = 0
        TP = 0
        FN = 0
        FP = 0

        print('')
        print('             *** EVALUATION RESULTS ***')
        print('')
        print('        Coverage by attack (positive classes)')
        classes = eval['Class'].unique()
        #Recall by class
        print('%6s %10s %10s' % ('FN','TP', 'recall'))
        for c in classes:
            if c != 'BENIGN':
                A = eval[(eval['prediction'] == True)  & (eval['Class'] == c)].shape[0]
                B = eval[(eval['prediction'] == False) & (eval['Class'] == c)].shape[0]

                print ( '%6d %10d %10.3f %26s' %(A, B, B / (A + B), c) )

                FN = FN + A     # cumulative FN
                TP = TP + B     # cumulative TP
            else:
                TN = eval[(eval['prediction'] == True)  & (eval['Class'] == 'BENIGN')].shape[0]
                FP = eval[(eval['prediction'] == False) & (eval['Class'] == 'BENIGN')].shape[0]

        print('%6s %10s' % ('----', '----'))
        print('%6d %10d %10s' % (FN, TP, 'total'))

        print('')
        print('Confusion matrix:')

        print('%42s' % ('prediction'))
        print('%36s | %14s' % (' | BENIGN (neg.)','ATTACK (pos.)'))
        print('       --------------|---------------|---------------')
        print('%28s  %6d | FP = %9d' % ('BENIGN (neg.) | TN = ', TN, FP))
        print('label  --------------|---------------|---------------')
        print('%28s  %6d | TP = %9d' % ('ATTACK (pos.) | FN = ', FN, TP))
        print('       --------------|---------------|---------------')

        recall = TP / (TP + FN)
        precision = 0
        if TP + FP != 0:
            precision = TP / (TP + FP)
        f1 = 0
        if precision + recall != 0:
            f1=2 * ((precision * recall) / (precision + recall))
        fpr = FP / (FP + TN)

        print('Metrics:')
        print('R = %5.3f  P = %5.3f  F1 score = %5.3f  FPR = %5.3f' % (recall, precision, f1, fpr))