fout=open("states.csv","a")
for line in open("states0.csv"):
    fout.write(line)
for num in range(1,3):
    f = open("states"+str(num)+".csv")
    for line in f:
         fout.write(line)
    f.close()
fout.close()

fout=open("labels.csv","a")
for line in open("labels0.csv"):
    fout.write(line)
for num in range(1,3):
    f = open("labels"+str(num)+".csv")
    for line in f:
         fout.write(line)
    f.close()
fout.close()