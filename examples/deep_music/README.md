# deep_music

This example tests the `deep` predictive learning model on predicting longer sequences with some structure that is not fully regular, by trying to predict the next note in a musical track.  This requires representing longer-term sequences.


# Mac playing the actual music as it goes

* open `Apple MIDI Setup`
* do `Window / Show MIDI Studio`
* double click on Default `IAC Driver`
* click `Device is online` -- the icon should now become undimmed
* open Garage Band (or Logic Pro if you have it)
* uncomment the Play setting in `deep_music.go` `ConfigEnv` method
* it should just work..

