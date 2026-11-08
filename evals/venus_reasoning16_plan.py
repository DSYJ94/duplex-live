"""Disjoint numeric-answer controls; references never enter inference messages."""
import json
from pathlib import Path

SYSTEM='Answer the question briefly. Give the final numeric result, without working or a list of alternatives.'


def build():
    examples=[
        ('There are 29 components. We use 11 and then receive 4. How many are there now?',29-11+4,'29 - 11 + 4'),
        ('A kit contains 41 screws. Four assemblies each use 6 screws. How many screws remain?',41-4*6,'41 - 4 * 6'),
        ('A cable is 2.7 metres long. We cut off 45 centimetres. How many centimetres remain?',270-45,'270 - 45'),
        ('A timer starts at minute 9 and runs for 22 minutes within the same hour. At which minute does it stop?',9+22,'9 + 22'),
        ('A triangle has sides of 8, 11 and 13 metres. What is its perimeter in metres?',8+11+13,'8 + 11 + 13'),
        ('A pump supplies 2.5 litres per minute for 6 minutes. How many litres does it supply?',15,'2.5 * 6'),
        ('Seven tickets cost 4 pounds each and there is one additional booking fee of 3 pounds. What is the total in pounds?',7*4+3,'7 * 4 + 3'),
        ('What is three eighths of 64?',3*64//8,'3 * 64 / 8'),
        ('A register starts at 8. Add 5, double the new value, then subtract 6. What is the final value?',(8+5)*2-6,'(8 + 5) * 2 - 6'),
        ('Container A has 17 litres and container B has 11 litres. Transfer 4 litres from A to B without loss. What is their combined volume in litres?',17+11,'17 + 11'),
        ('Mina has 18 tokens. Sol has 5 fewer. Ren has three times as many as Sol. How many tokens does Ren have?',(18-5)*3,'(18 - 5) * 3'),
        ('A delivery was scheduled for day 16. It is delayed by 5 days, then brought forward by 2 days. On which day will it arrive?',16+5-2,'16 + 5 - 2'),
        ('A resistor has 15 volts across it and carries 3 amperes. Using V = I times R, what is its resistance in ohms?',15//3,'15 / 3'),
        ('An object moves 90 metres in 6 seconds at constant speed. What is its speed in metres per second?',90//6,'90 / 6'),
        ('A stack receives 7, then 12, then 19. Two last-in-first-out pops occur. Which number remains?',7,'[7, 12, 19] -> pop 19 -> pop 12 -> 7'),
        ('What is the bitwise AND of the nonnegative integers 13 and 6, expressed in decimal?',13&6,'0b1101 & 0b0110 = 0b0100'),
    ]
    return dict(schema='dl-authored-numeric16-v1',system=SYSTEM,cases=[
        dict(id='numeric16-%02d'%i,question=q,expected=str(n),reference_derivation=d)
        for i,(q,n,d) in enumerate(examples)])


def messages(plan,row):
    return [dict(role='system',content=plan['system']),dict(role='user',content=row['question'])]


if __name__=='__main__':
    Path('evals/fixtures/venus-numeric16-text-20261108.json').write_text(json.dumps(build(),indent=2)+'\n')
