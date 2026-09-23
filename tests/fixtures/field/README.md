# Field frames

Real frames from the phone, not renders. They are the regression set for
finding cards the way a shop shows them: on a light wood counter at 20-40
degrees, sleeved, several to a frame, overlapping, a thumb over an edge,
motion-soft.

* `desk_*_inset.jpg` are the exact 540x632 frames the server received during
  a live session (read out of the app's debug inset in the owner's
  screenshots of 2026-09-23, v2.16.0). The top 6% was under the inset's text
  strip and is inpainted.
* `hand_ar*.jpg` are the live preview from earlier screenshots, with the HUD,
  the viewfinder guide lines and border inpainted, scaled to 540x632.

`annotations.json` holds the card outlines (by eye, about +/-3 px) and what
each frame shows. On these frames the contour detector alone
(`geometry.find_card_quad`) found the card under the reticle in 0 of 17
checks; `tests/test_field.py` holds the scene search to what it achieves now.

Add frames here when the reader fails in a shop: the debug inset shows the
exact frame the server got, so a screenshot of a failure is a test case.
