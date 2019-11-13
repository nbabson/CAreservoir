# CAreservoir

## overview

This C++ project implements a Cellular Automaton reservoir (ReCA). At present 
the reservoir is tested on the 5-bit memory task benchmark, which is 
the standard benchmark used to evalulate ReCAs. The reservoir weights are 
trained either with linear regression using the AlgLib library, or support
vector machines using the free software SVMTorch and SVMTest. Command line options allow
the user to set the number of states, number of CA iterations, reservoir size,
and number of subreservoirs. The draw option saves the CA to ca.txt and draws
the first WIDTH iterations of the CA to ca.ppm. There is also the option to use
two rules to generate a non-uniform CA reservoir. The -bf option stochastically
searches for promising 3 state rules, saving  the best to three_state_rules.txt
(the rulefile command line argument is ignored when -bf is set). 

The file rule.txt contains a list of 3 state rules that work as reservoirs for
the 5-bit task.

rule<#>.txt files contain the most successful elementary (2 state) CA rules.

The repository includes a number of compiled .o files that are part of the
AlgLib library.

## usage

You can clone the project using
```
git clone https://github.com/nbabson/CAreservoir
```
Type 
```
make
```
to build the project, and
```
./CAreservoir
```
to run it. You will be shown a usage message listing a range of options.

## license

This project is covered by the MIT license.
