// Shared by both decks. Fill in the names once, here.
//
// Pair A works without GenAI in Phase I and with it in Phase II; pair B the other way round.
// Teammates' student IDs: K12518337, K12542264, K12315344.

#import "template/slides-template.typ": todo

#let group = [Azure Beetles]
#let course = [UE AI and Visualization, JKU, 2026W]
#let institution = [Visual Data Science Lab]
#let title = [How a cone detector sees distance and motion]

#let pair-a = ([Riccardo Persello], [Mark Berezc])
#let pair-b = ([Eram Shaikh], [Arteljana Saliasi])

#let names(pair) = pair.join[ and ]
#let everyone = (pair-a + pair-b).join[, ]
