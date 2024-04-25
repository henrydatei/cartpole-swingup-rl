# cartpole-swingup-rl

25.4.2024
- 10:53 speed von 60k zu 55k, weil cart immer angestoßen ist, das ist bei 50k nicht passiert, vielleicht sind da ungenauigkeiten in der position bei höheren geschwindigkeiten -> maximale geschwindigkeit, wo es nicht zu solchen problemen kommt. Auch wieder 9 umdrehungen bis zum ende, das wurde ja auch im arduino so implementiert => funktioniert nicht, reset fährt nicht zur mitte, sondern verschiebt sich immer weiter an rand
- 11:01 speed zu 50k => bringt auch nix, vielleicht zu hoher Strom (2A RMS; 2.83A Peak)?
- 11:05 60k speed, 1.51A RMS, 2.14A Peak => hilft auch nicht