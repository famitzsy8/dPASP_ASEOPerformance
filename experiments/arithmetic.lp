% Choosing the N-arity of our number system
#const n = 24.

{d(D, 0..n)} = 1 :- digit(D).

0.5::odd(D) :- d(D, 1). % Here we put a probability we will not use in order for ASEO to work
odd(D) :- d(D, 3).
odd(D) :- d(D, 5).
odd(D) :- d(D, 7).
odd(D) :- d(D, 9).

% Creating 3 digits
digit(1).
digit(2).
digit(3).

% And all the possible combinations a number can be created
expr(N) :- d(1, A), d(2, B), d(3, C), N = 625 * A + 25 * B + C.
expr(N) :- d(1, A), d(2, B), d(3, C), N = 625 * A + 25 * C + B.
expr(N) :- d(1, A), d(2, B), d(3, C), N = 625 * B + 25 * C + A.
expr(N) :- d(1, A), d(2, B), d(3, C), N = 625 * B + 25 * A + C.
expr(N) :- d(1, A), d(2, B), d(3, C), N = 625 * C + 25 * A + B.
expr(N) :- d(1, A), d(2, B), d(3, C), N = 625 * C + 25 * B + A.

% We want the probability that the generated number is (strictly) between 100 and 5900
valid_expr :- expr(X), X < 5900, X > 100.

#query(valid_expr).

#semantics maxent.