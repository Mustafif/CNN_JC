#let definition(title: none, body) = {
  block(
    width: 100%,
    inset: 10pt,
    fill: luma(240),
    radius: 4pt,
    [
      #text(weight: "bold")[Definition: #title]

      #body
    ]
  )
}

#let todo(body) = {
block(
width: 100%,
inset: 10pt,
fill: yellow,
radius: 4pt,

[
  #text(weight: "bold")[TODO]

  #body
]
)
}


#let note(body) = {
block(
  width: 100%,
  inset: 10pt,
  fill: rgb(173, 216, 230),
  radius: 4pt,
  [
    #text(weight: "bold")[NOTE:]

    #body
  ]
)
}
