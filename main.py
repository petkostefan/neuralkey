from tree_parity_machine import TreeParityMachine
import numpy as np
import time
import sys
import matplotlib.pyplot as plt


# Hiperparametri mreže
k = 100
n = 10
l = 10

# Pravila učenja
learning_rules = ['hebbian', 'anti_hebbian', 'random_walk']
learning_rule = learning_rules[0]

# Generator vektora nasumičnih brojeva
def random_vector():
    return np.random.randint(-l, l+1, [k, n])

# Funkcija za evaluaciju sinhronizacije dve mašine
def sync_score(m1, m2):
    return 1.0 - np.average(1.0 * np.abs(m1.W - m2.W)/(2 * l))


# Kreiranje 3 mašine: Alice, Bob, Eve. Eve će pokušati da presretne komunikaciju između Alice i Boba
print(f'Kreiranje mašina sa parametrima: k={k}, n={n}, l={l}')
print(f'Korišćenjem {learning_rule} pravila učenja:')

Alice = TreeParityMachine(k, n, l)
Bob = TreeParityMachine(k, n, l)
Eve = TreeParityMachine(k, n, l)

# Sinhronizacija težina
sync = False 
updates_cnt = 0
eve_updates_cnt = 0
sync_history = []
eve_history = []
start_time =  time.time()

while(not sync):

    X = random_vector() # Kreiranje vektora dimenzija [k, n] sa nasumičnim vrednostima.

    tauA = Alice.get_output(X)
    tauB = Bob.get_output(X)
    tauE = Eve.get_output(X)

    Alice.update(tauB, learning_rule) # Ažuriranje Alice-ine mašine Bobovim izlazom
    Bob.update(tauA, learning_rule) # Ažuriranje Bobove mašine Alice-inim izlazom

    # Eve ažurira vrednost samo ako važi uslov: tauA == tauB == tauE
    if tauA == tauB == tauE:
        Eve.update(tauA, learning_rule)
        eve_updates_cnt += 1

    updates_cnt += 1

    score = 100 * sync_score(Alice, Bob) # Računanje nivoa sinhronizacije Alice-ine i Bobove mašine
    eve_score = 100 * sync_score(Alice, Eve) # Računanje nivoa sinhronizacije Alice-ine i Eve-ine mašine
    sync_history.append(score) # Čuvanje rezultata za ksnije plotovanje
    eve_history.append(eve_score)

    sys.stdout.write('\r' + "Sinhronizacija = " + str(int(score)) + "% / Broj iteracija = " + str(updates_cnt) + " / Broj ažuriranja Eve = " + str(eve_updates_cnt) + " / Broj ažuriranja pravila učenja :" +str(Alice.update_count))

    # Ako je rezultat 100, mašine su sinhronizovane
    if score == 100:
        sync = True

end_time = time.time()
time_taken = end_time - start_time

# Ispisivanje rezultata
print('\nMašine su se uspešno sinhronizovale.')
print(f'Proces je trajao {time_taken:.2f} sekundi.')
print(f'Broj iteracija: {updates_cnt}.')
print(f'Procenat ažuriranja mašina u odnosu na broj iteracija: {float(Alice.update_count / updates_cnt)*100:.2f}%')

# Provera rezultata Eve
if eve_score >= 100:
    print('Eve je uspela da sinhronizuje svoju mašinu sa Alice i Bobom.')
else:
    print(f'Eve je uspela da sinhronizuje svoju mašinu {str(int(eve_score))}% i uradila je {eve_updates_cnt} ažuriranja.')

# Crtanje grafika sinhronizacije
plt.plot(sync_history)
plt.plot(eve_history)
plt.title("Grafik toka sinhronizacije mašina")
plt.xlabel("Broj iteracija")
plt.ylabel("Nivo sinhronizacije")
plt.show()