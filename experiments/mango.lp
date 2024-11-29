% Here we set the probabilities that a certain seller sells a specific fruit
0.2::sells(mango, 1). 0.99999::sells(apple, 1).
0.4::sells(mango, 2). 0.6::sells(apple, 2).
0.7::sells(mango, 3). 0.3::sells(apple, 3).

% Declaring the mango buyers
mango_buyer(1..4).

% The condition that a deal happens
0.5::deal(X, Y) :- mango_buyer(X), sells(mango, Y), not sells(apple, Y).

% All the possible deals (except for (4, 1) and (4, 2) to control program size)
was_deal :- deal(1, 1).
was_deal :- deal(1, 2).
was_deal :- deal(1, 3).
was_deal :- deal(2, 1).
was_deal :- deal(2, 2).
was_deal :- deal(2, 3).
was_deal :- deal(3, 1).
was_deal :- deal(3, 2).
was_deal :- deal(3, 3).
was_deal :- deal(4, 3).

#semantics maxent.

#query(was_deal).
