import sys, random

if len(sys.argv) < 2:
	print 'Usage: python gen_data.py num_of_feature num_of_data'
	sys.exit()

nf = int(sys.argv[1])
nd = int(sys.argv[2])

header = ['id']
for i in range(nf):
	header.append('feature%d' % (i+1))
header.append('label')
print ','.join(header)

for i in range(nd):
	row = [i]
	for j in range(nf):
		row.append(random.uniform(-1,1))
	row.append(random.choice([1.0,0.0]))
	print ','.join([str(x) for x in row])

