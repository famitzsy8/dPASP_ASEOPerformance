% Define digits and positions
digit(1..3).
pos(0..8).

% Define rows and columns using position-to-coordinate mapping
row(P, R) :- pos(P), R = P / 3.
col(P, C) :- pos(P), C = P \ 3.

% Choice rule: For each position, assign exactly one digit
{ assign(P, D) : digit(D) } = 1 :- pos(P).

% Row constraint: No repeated digits in the same row
inconsistent_rows :- assign(P1, D), assign(P2, D), row(P1, R), row(P2, R), P1 != P2.

% Column constraint: No repeated digits in the same column
inconsistent_cols :- assign(P1, D), assign(P2, D), col(P1, C), col(P2, C), P1 != P2.
0.5::fail :- inconsistent_cols, inconsistent_rows.

#semantics maxent.

#query fail.