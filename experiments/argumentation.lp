person(1). person(2).

% Probabilities that a certain person will use a certain argument after having said another
0.7::argues(1, 1) :- person(1), argues(2, 1).
0.4::argues(1, 1) :- person(1), argues(7, 1).
0.75::argues(2, 1) :- person(1), argues(4, 2).
0.9::argues(2, 1) :- person(1), argues(7, 1).
0.9::argues(5, 1) :- person(1), argues(2, 1).
0.95::argues(6, 1) :- person(1), argues(1, 2).
0.4::argues(3, 1) :- person(1), argues(1, 1).
0.1::argues(3, 1) :- person(1), argues(2, 1).

% The exact same probabilities hold for the other person
0.7::argues(1, 2) :- person(2), argues(2, 2).
0.1::argues(1, 2) :- person(2), argues(7, 2).
0.4::argues(2, 2) :- person(2), argues(4, 1).
0.9::argues(2, 2) :- person(2), argues(7, 2).
0.3::argues(5, 2) :- person(2), argues(2, 2).
0.7::argues(6, 2) :- person(2), argues(1, 1).
0.3::argues(3, 2) :- person(2), argues(1, 2).
0.6::argues(3, 2) :- person(2), argues(2, 2).

% Initial arguments the persons use to enter the debate
argues(4, 1). argues(7, 2).

% A person constructs a fact in a debate when using a certain subset of possible arguments
fact(P) :-  argues(6, P).

% A person wins if he has constructed a fact whereas the other person didn't
wins(1, 2) :- fact(1), not fact(2).
wins(1, 2) :- fact(1), fact(2).
wins(2, 1) :- not fact(1), fact(2).
wins(2, 1) :- not fact(1), not fact(2).

#semantics maxent.

#query(wins(2, 1)).