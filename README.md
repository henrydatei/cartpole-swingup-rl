# cartpole-swingup-rl

25.4.2024
- 10:53 speed von 60k zu 55k, weil cart immer angestoßen ist, das ist bei 50k nicht passiert, vielleicht sind da ungenauigkeiten in der position bei höheren geschwindigkeiten -> maximale geschwindigkeit, wo es nicht zu solchen problemen kommt. Auch wieder 9 umdrehungen bis zum ende, das wurde ja auch im arduino so implementiert => funktioniert nicht, reset fährt nicht zur mitte, sondern verschiebt sich immer weiter an rand
- 11:01 speed zu 50k => bringt auch nix, vielleicht zu hoher Strom (2A RMS; 2.83A Peak)?
- 11:05 60k speed, 1.51A RMS, 2.14A Peak => hilft auch nicht
- 11:13, 0.86A Peak, 0.61A RMS => immer noch ungenau, es muss am Motor liegen
- 11:26-11:33 Motor ausgetauscht, 2.14A Peak, 1.51A RMS, Motor verträgt 1.8A Nennstrom => immerhin jetzt genauer
- 13:07 Kühlung an Motor angebracht, Verbesserung bezüglich Reward-Funktion zu erkennen, 2048 Steps/Episode und insgesamt 50k Steps
- 16:51-59 Reward Clipping wenn Winkelgeschwindigkeit zu hoch, Bestafung für Position zu weit von der Mitte entfernt, Training bis 30.4. (etwa 4 Tage, 345000 Sekunden). Aktuelle Geschwindigkeit: 51200 Steps in 11706 Sekunden => 1.5 Millionen Steps