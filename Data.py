'''
gestion des donnés et mise en forme
'''

import numpy as np
from os import listdir
from os.path import isfile, join
import csv
import random



def get_Spara(file):    #return all s paramters in file
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        linecount = 0
        s11 = []
        s21 = []
        s22 = []
        s31 = []
        s32 = []
        s33 = []
        s41 = []
        s42 = []
        s43 = []
        s44 = []
        for row in reader:
            if linecount == 0:
                linecount += 1
            else:
                s11.append(float(row[5]))
                s21.append(float(row[6]))
                s22.append(float(row[7]))
                s31.append(float(row[8]))
                s32.append(float(row[9]))
                s33.append(float(row[10]))
                s41.append(float(row[11]))
                s42.append(float(row[12]))
                s43.append(float(row[13]))
                s44.append(float(row[14]))
    return s11,s21,s22,s31,s32,s33,s41,s42,s43,s44

def rotated(array_2d):
    list_of_tuples = zip(*array_2d[::-1])
    return [list(elem) for elem in list_of_tuples]
    
def reverse(table):
    table1 = []
    for line in table:
        table1.append(list(reversed(line)))
    return table1
    
def get_patch(file): #return patch patern in file and also rotated and reversed version
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        table = []
        table1 = []
        for row in reader:
            for i in range(int(len(row)/32)):
                res = [eval(k) for k in row[int(i*32):int((i+1)*32)]]
                table1.append(list(reversed(res)))
                table.append(res)
    return table, table1, rotated(table),reverse(rotated(table)),rotated(rotated(table)),reverse(rotated(rotated(table))),rotated(rotated(rotated(table))),reverse(rotated(rotated(rotated(table))))

def get_data(augmented):
    #input
    name=[]
    tab_in=[]
    fichiers_in = [f for f in listdir("patch") if isfile(join("patch", f))]
    for f in fichiers_in:
        name.append(f)
        tab,tab1,tab2,tab3,tab4,tab5,tab6,tab7=get_patch("patch\\"+f)
        tab_in.append(tab)
        if augmented:
            name.append(f)
            tab_in.append(tab1)
            name.append(f)
            tab_in.append(tab2)
            name.append(f)
            tab_in.append(tab3)
            name.append(f)
            tab_in.append(tab4)
            name.append(f)
            tab_in.append(tab5)
            name.append(f)
            tab_in.append(tab6)
            name.append(f)
            tab_in.append(tab7)
    #output
    tab_out = []
    fichiers_out = [f for f in listdir("data") if isfile(join("data", f))]
    for f in fichiers_out:
        s11,s21,s22,s31,s32,s33,s41,s42,s43,s44=get_Spara("data\\"+f)
        data=[0]*1010
       #tab
        for i in range(len(s21)):
            data[i]=s11[i]
            data[i+101]=s21[i]   
            data[i+202]=s22[i]
            data[i+303]=s31[i]  
            data[i+404]=s32[i]  
            data[i+505]=s33[i]  
            data[i+606]=s41[i]  
            data[i+707]=s42[i]  
            data[i+808]=s43[i]  
            data[i+909]=s44[i]  
             
        tab_out.append(data)
        if augmented:
            #tab1
            for i in range(len(s21)):
                data[i]=s22[i]
                data[i+101]=s21[i]   
                data[i+202]=s11[i]
                data[i+303]=s42[i]  
                data[i+404]=s41[i]  
                data[i+505]=s44[i]  
                data[i+606]=s32[i]  
                data[i+707]=s31[i]  
                data[i+808]=s43[i]  
                data[i+909]=s33[i]  
            tab_out.append(data)
            #tab2
            for i in range(len(s21)):
                data[i]=s44[i]
                data[i+101]=s41[i]   
                data[i+202]=s11[i]
                data[i+303]=s42[i]  
                data[i+404]=s21[i]  
                data[i+505]=s22[i]  
                data[i+606]=s43[i]  
                data[i+707]=s31[i]  
                data[i+808]=s32[i]  
                data[i+909]=s33[i]  
            tab_out.append(data)
            #tab3
            for i in range(len(s21)):
                data[i]=s11[i]
                data[i+101]=s41[i]   
                data[i+202]=s44[i]
                data[i+303]=s31[i]  
                data[i+404]=s43[i]  
                data[i+505]=s33[i]  
                data[i+606]=s21[i]  
                data[i+707]=s42[i]  
                data[i+808]=s32[i]  
                data[i+909]=s22[i]  
            tab_out.append(data)
            #tab4
            for i in range(len(s21)):
                data[i]=s33[i]
                data[i+101]=s43[i]   
                data[i+202]=s44[i]
                data[i+303]=s31[i]  
                data[i+404]=s41[i]  
                data[i+505]=s11[i]  
                data[i+606]=s32[i]  
                data[i+707]=s42[i]  
                data[i+808]=s21[i]  
                data[i+909]=s22[i]  
            tab_out.append(data)
            #tab5
            for i in range(len(s21)):
                data[i]=s44[i]
                data[i+101]=s43[i]   
                data[i+202]=s33[i]
                data[i+303]=s42[i]  
                data[i+404]=s32[i]  
                data[i+505]=s22[i]  
                data[i+606]=s41[i]  
                data[i+707]=s31[i]  
                data[i+808]=s21[i]  
                data[i+909]=s11[i]  
            tab_out.append(data)
            #tab6
            for i in range(len(s21)):
                data[i]=s22[i]
                data[i+101]=s32[i]   
                data[i+202]=s33[i]
                data[i+303]=s42[i]  
                data[i+404]=s43[i]  
                data[i+505]=s44[i]  
                data[i+606]=s21[i]  
                data[i+707]=s31[i]  
                data[i+808]=s41[i]  
                data[i+909]=s11[i]  
            tab_out.append(data)
            #tab7
            for i in range(len(s21)):
                data[i]=s33[i]
                data[i+101]=s32[i]   
                data[i+202]=s22[i]
                data[i+303]=s31[i]  
                data[i+404]=s21[i]  
                data[i+505]=s11[i]  
                data[i+606]=s43[i]  
                data[i+707]=s42[i]  
                data[i+808]=s41[i]  
                data[i+909]=s44[i]  
            tab_out.append(data)

    return tab_in, tab_out,name


def shuffle_data(datain,dataout,name):
    if len(datain) != len(dataout):
        raise "Problem length of input and labelled data not equal"
    datain_shuffled=[0]*len(datain)
    dataout_shuffled=[0]*len(datain)
    name_shuffled=[0]*len(datain)
    index=list(range(len(datain)))
    index=random.sample(index,len(index))
    for i in range(len(index)):

        datain_shuffled[i]=datain[index[i]]
        dataout_shuffled[i]=dataout[index[i]]
        name_shuffled[i]=name[index[i]]
    return datain_shuffled, dataout_shuffled,name_shuffled

def save(data,filename = "savedfile.csv"):
    with open(filename, 'w+',newline ='') as csvfile:   
        csvwriter = csv.writer(csvfile)
        for i  in range(len(data)):
            csvwriter.writerows([data[i]])


#debug part
if __name__ == '__main__':
    t1, t2,t3=get_data(True)

    print(len(t1))
    print(len(t2))
    print(len(t3))