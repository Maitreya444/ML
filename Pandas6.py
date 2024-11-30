import os
import fnmatch
from sys import *
import xlsxwriter

def ExcelCreate(name):
    workbook = xlsxwriter.Workbook(name)

    worksheet = workbook.add_worksheet()

    worksheet.write('A1', 'Name')
    worksheet.write('B1', 'College')
    worksheet.write('C1', 'MailID')
    worksheet.write('D1', 'Mobile')

    workbook.close()

def main():
    print("------Maitreya Gangurde------")

    print("Application Name :"+ argv[0])

    if(len(argv) !=2):
        print("Error : Invalid number of arguments")
        exit()

    if (argv[1] == "-h") or (argv[1] == "-H"):
        print("This Script is used to create excel file and write data into it")
        exit()

    if (argv[1] == "-u") or (argv[1]== "-U"):
        print("Usage : ApplicationName Name_Of_File")
        exit()

    try:
        ExcelCreate(argv[1])

    except Exception:
        print("Error : Invalid Input")

if __name__ =="__main__":
    main()