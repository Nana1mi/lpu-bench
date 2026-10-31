/**
 * mpu_add_u32.m.asm
 *
 * Created on: 2024-08-15
 *      Author: tianxinli
 */
.section .text.m0, "ax"
.file	"mpu_add_u32.m.asm"
// DO NOT MODIFY THE CONTENT ABOVE
.global	mpu_add_u32
mpu_add_u32:
	// Do not modify the value of KI15

	R0:M[0]->BIU.T0 || R1:M[1]->BIU.T4 || R2:M[2]->BIU.T8;
    NOP ;
    NOP ;
	NOP ;

	BIU0: wait 0 				 || BIU1: wait 0 				 || SHU0: wait 13 			  || BIU2: wait 16;
LOOP:
    BIU0: LOAD(T0)(A++)->SHU0.T0 || BIU1: LOAD(T4)(A++)->SHU0.T1 || SHU0: T0+T1(T)(U)(I32)->BIU.T3 || BIU2: Store(T3,T8)(A++) || MFETCH: IF ( 1 != KI0 ) JUMP %LOOP(DECKI);

	BIU0: wait 0 || BIU1: wait 0 || SHU0: wait 0  || BIU2: wait 0 || MFETCH:MPUWAIT 0;

MFETCH: JUMP KI15;



