# -*- coding: utf-8 -*-
from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Mutators
from pyevolve import Initializators
from pyevolve import GAllele
import pyevolve
import os

# This function is the evaluation function, we want
# to give high score to more zero'ed chromosomes
def eval_func(c):
	print c
	name = os.popen('python train2.py --embedding_dim='+str(c[0])+' --filter_sizes='+str(c[1])+','+str(c[2])+','+str(c[3]) \
	+' --num_filters='+ str(c[4]) + ' --dropout_keep_prob='+ str(c[5]/100.0)+' --l2_reg_lambda='+ str(c[6]/100.0) \
	+' --batch_size='+ str(c[7])+' --num_epochs='+str(c[8])).read()
	print '--------'
	n = name.split('/')
	n = int(n[len(n)-1])	
	acc, prec, rec, f1 = os.popen('./eval.py --eval_train --checkpoint_dir="./runs/'+str(n)+'/checkpoints/"').read().split(' ')
	print acc, prec, rec, f1, 'Iteração: ', n
	return (float(acc) + float(prec) + float(rec) + float(f1))

pyevolve.logEnable()

setOfAlleles = GAllele.GAlleles()
setOfAlleles.add(GAllele.GAlleleRange(64, 256))
setOfAlleles.add(GAllele.GAlleleRange(2, 10))
setOfAlleles.add(GAllele.GAlleleRange(2, 10))
setOfAlleles.add(GAllele.GAlleleRange(2, 10))
setOfAlleles.add(GAllele.GAlleleRange(10, 256))
setOfAlleles.add(GAllele.GAlleleRange(1, 100))
setOfAlleles.add(GAllele.GAlleleRange(0, 100))
setOfAlleles.add(GAllele.GAlleleRange(64, 256))
setOfAlleles.add(GAllele.GAlleleRange(100, 500))


genome = G1DList.G1DList(9)
genome.setParams(allele=setOfAlleles)

# The evaluator function (objective function)
genome.evaluator.set(eval_func)

# This mutator and initializator will take care of
# initializing valid individuals based on the allele set
# that we have defined before
genome.mutator.set(Mutators.G1DListMutatorAllele)
genome.initializator.set(Initializators.G1DListInitializatorAllele)

# Genetic Algorithm Instance
ga = GSimpleGA.GSimpleGA(genome)
ga.setGenerations(5)

# Do the evolution, with stats dump
# frequency of 10 generations
ga.evolve(freq_stats=5)

# Best individual
print ga.bestIndividual()
