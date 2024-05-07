# cartpole-swingup-rl

25.4.2024
- 10:53 speed von 60k zu 55k, weil cart immer angestoßen ist, das ist bei 50k nicht passiert, vielleicht sind da ungenauigkeiten in der position bei höheren geschwindigkeiten -> maximale geschwindigkeit, wo es nicht zu solchen problemen kommt. Auch wieder 9 umdrehungen bis zum ende, das wurde ja auch im arduino so implementiert => funktioniert nicht, reset fährt nicht zur mitte, sondern verschiebt sich immer weiter an rand
- 11:01 speed zu 50k => bringt auch nix, vielleicht zu hoher Strom (2A RMS; 2.83A Peak)?
- 11:05 60k speed, 1.51A RMS, 2.14A Peak => hilft auch nicht
- 11:13, 0.86A Peak, 0.61A RMS => immer noch ungenau, es muss am Motor liegen
- 11:26-11:33 Motor ausgetauscht, 2.14A Peak, 1.51A RMS, Motor verträgt 1.8A Nennstrom => immerhin jetzt genauer
- 13:07 Kühlung an Motor angebracht, Verbesserung bezüglich Reward-Funktion zu erkennen, 2048 Steps/Episode und insgesamt 50k Steps
- 16:51-59 Reward Clipping wenn Winkelgeschwindigkeit zu hoch, Bestafung für Position zu weit von der Mitte entfernt, Training bis 30.4. (etwa 4 Tage, 345000 Sekunden). Aktuelle Geschwindigkeit: 51200 Steps in 11706 Sekunden => 1.5 Millionen Steps => Kamera hat nach 1.3 Millionen Steps nicht mehr funktioniert, es wurden keine Winkel mehr verarbeitet. Neustart der Kamera hat geholfen, Laut Ostap und Ankit soll es wohl zwischendrin ganz gut ausgesehen haben, ironischerweise da, wo der Reward besonders negativ war... Aber zumindest blieb der Cartpole in der Mitte, Pentalty für zu weit außen hat funktioniert

30.04.2024
- 10:56: Ankits Reward Funktion mit 5 Winkeln => sieht eigentlich ganz gut aus, in 8000 timesteps (4 Episoden) hat es das Pendel schon nach oben geschafft
- 12:47: 3 actions, less exploring, Abnutzungserscheinungen => Probleme mit dem Export der Obervations etc.
- 13:54-57: Test ob Probleme mit Export behoben, aufräumen des Kabelsalates => irgendwie läuft es jetzt nicht mehr, der Motor bewegt sich nicht mehr und der Lüfter für den Motor läuft nur schleppend an.
- 2:24-56: Lüfter für Motor an extra 5V, Lüfter für Rasperry an 3.3V (beides Raspi Extension Board), sonst viel rumprobiert, anderen stepper motor (2A variante) aber Strom auf 1.69A Peak und 1.2A RMS für weniger Hitze. Pole kommt trotzdem nach oben
- 3:42: Training für 600k steps starten. sollte bis zum 2.5. laufen => nach ca 250k steps passiert ziemloch wenig, das cart bleibt zwar in der mitte, unternimmt aber keine versuche mehr das pendel nach oben zu bekommen (vielleicht ein Problem mit dem angular velocity penalty? immerhin nimmt diese ab im verlauf der zeit)

2.5.2024
- 13:08: Fehler in der Berechnug der Angular Velocity behoben, Reward Funktion bestraft Angle Velocity nur, wenn letzter (aka aktuellster) Winkel unter 12° ist, Position Penalty jetzt nur halb so stark, Test mit 10k Steps, 2 Actions (Action 0, die nichts macht, kann man ignorieren) => es scheint durchzulaufen, wirkliches Lernen ist schwierig nach so kurzer Zeit und Winkelgeschwindigkeitsbestimmung ist immer noch nicht richtig
- 14:22-24: Fehler behoben, hoffentlich diesmal richtig, Zeit von Aufnahme des Bildes bis Verwendung des Winkels -> Delay bestimmen, Pole_up in Observation hinzugefügt => Pole_up und angular_velocity scheinen vertauscht zu sein
- 15:12: Richtigstellung der Vertauschung
- 16:05-12: mal laufen lassen mit 1.5m steps, sollte jetzt auch nur außerhalb der arbeitszeiten laufen (8:30-18:30 werktags) => kein gutes Ergebnis, Kamera ist nach 1.3m steps nicht mehr gelaufen, simples hin- und herfahren in der Mitte. Nach etwa 200k steps blieb die mean_ep_length auch konstant bei 2048, mean_ep_reward bei rund 750. Das ist rund 2048 * exp(cos(170)), da der Winkel rund 170° Grad ist, bei dem leichten hin- und herfahren.