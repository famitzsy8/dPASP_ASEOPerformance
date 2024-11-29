% All the persons in our tiny little world
person(anna).
person(bob).
person(peter).
person(larry).
person(steve).
person(anthony).
person(gwen).

% The persons who smoke
smokes(anna).
smokes(peter).
smokes(steve).
smokes(gwen).

% The social network
0.3::befriends(anna, bob).
0.8::befriends(peter, bob).
0.9::befriends(larry, peter).
0.2::befriends(peter, anna).
0.5::befriends(steve, anthony).
0.5::befriends(anthony, larry).
0.2::befriends(gwen, larry).

% Generally, pious non-smokers can succumb to the cigarette when the stress hits them
0.4::stressed(X) :- person(X).
smokes(X) :- stressed(X).

% And also if they have a friend who has nothing against a smoke or two
0.9::smokes(X) :- befriends(Y,X), smokes(Y).

#semantics maxent.

#query(smokes(bob), smokes(larry)).