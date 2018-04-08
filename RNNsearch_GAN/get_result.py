fd=open('NIST.result','r')
output = fd.readlines()
BLEUStrIndex = output.index('BLEU score = ')
blu_new = float(output[BLEUStrIndex+13:BLEUStrIndex+19])


