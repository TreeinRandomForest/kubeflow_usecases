import matplotlib.pylab as plt
plt.ion()

fig = plt.figure(figsize=(10,10)) 
ax = fig.subplots()
A = [1,2,3,4]
B = [1,2,3]

for a in A:
    for b in B:
        plt.plot([a], [b], 'p', c = 'b')

plt.xlabel('Hyperparameter A')
plt.ylabel('Hyperparameter B')
plt.xticks(A)
plt.yticks(B)
ax.set_xticklabels([f'$a_{a}$' for a in A])
ax.set_yticklabels([f'$b_{b}$' for b in B])
plt.title('Grid for hyperparameter sets to search over')
plt.grid()

yoffset = 0.01
xoffset = 0.01
for a in A:
    for b in B:
        train_acc = 0.9 - (abs(a-2.7)**2)*(abs(b-1.7)**2)*0.10
        val_acc = train_acc * 0.95
        plt.text(a + xoffset, b + yoffset, f"Train Acc. = {train_acc:.3f} \n" + r"$\bf{Val. Acc}$" + f" = {val_acc:.3f}")
plt.savefig('Ch4_hyperparam_grid.png')