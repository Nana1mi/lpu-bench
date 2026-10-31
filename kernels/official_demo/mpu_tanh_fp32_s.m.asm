/**
 * mpu_tanh_fp32_s.m.asm
 *
 * Created on: 2025-07-19 10:04:43
 *      Author: wenzixuan
 */
 .section .text.m0, "ax"
 .file	"mpu_tanh_fp32_s.m.asm"
 // DO NOT MODIFY THE CONTENT ABOVE
 
 .global mpu_tanh_fp32
 
 mpu_tanh_fp32:
     R0:M[3]->BIU.T3;	// constant
     R0:M[0]->BIU.T0;	// input
     R0:M[1]->BIU.T1;	// table_0
     R0:M[2]->BIU.T2;	// table_1
     R3:M[64]->BIU.T12;	// output
 
     BIU0:KG(T3[0])->SHU3.T7;  // 0x80000000, keep sign bit
     BIU0:KG(T3[1])->SHU3.T6;  // 0x7fffffff, turn sign bit to 0
 
     BIU0:KG(T3[2])->FMA2.T1;  // 0x0319999a
     BIU0:KG(T3[2])->FMA0.T3;  // 0x0319999a
     BIU0:KG(T3[3])->FMA2.T3;  // 0x41077800
 
     // a0                                  || a1                                       || a2                                       || a3
     // a4
     BIU0:KG(T3[4])->FMA1.T0                || BIU2:KG(T3[5])->FMAC3.T0                 || BIU1:KG(T3[6])->FMAC2.T0                 || BIU3:KG(T3[7])->FMA3.T1;
     BIU0:KG(T3[8])->FMA3.T2;
     FMA1:0+T0*1(shiftmode0)(FP32)->FMA1.T8 || FMAC3:0+T0*1(shiftmode0)(FP32)->FMAC3.T8 || FMAC2:0+T0*1(shiftmode0)(FP32)->FMAC2.T8;
     FMA3:0+T1*1(shiftmode0)(FP32)->FMA3.T8;
     FMA3:0+T2*1(shiftmode0)(FP32)->FMA3.T9;
 
     // -start/step                          || 1/step   
     // read start                           || read step                                
     BIU0:KG(T3[9])->FMA3.T3                 || BIU1:KG(T3[11])->FMAC0.T0;
     BIU0:KG(T3[10])->FMA3.T4                || BIU1:KG(T3[12])->FMAC0.T1;
     FMA3:0+T3*1(shiftmode0)(FP32)->FMA3.T10 || FMAC0:0+T0*1(shiftmode0)(FP32)->FMAC0.T10;
     FMA3:0+T4*1(shiftmode0)(FP32)->FMA3.T11 || FMAC0:0+T1*1(shiftmode0)(FP32)->FMAC0.T11;
 
     // read port config
     // read x                      || read x^2                       || read x^2                       || read y0                        || read x                         || read y1                        || read x^2
     R0:ConfigMC(0)->MC.R(IStart)   || R1:ConfigMC(64)->MC.R(IStart)  || R2:ConfigMC(64)->MC.R(IStart)  || R3:ConfigMC(96)->MC.R(IStart)  || R4:ConfigMC(0)->MC.R(IStart)   || R6:ConfigMC(160)->MC.R(IStart) || R7:ConfigMC(64)->MC.R(IStart);
     R0:ConfigMC(1)->MC.R(IStep)    || R1:ConfigMC(1)->MC.R(IStep)    || R2:ConfigMC(1)->MC.R(IStep)    || R3:ConfigMC(1)->MC.R(IStep)    || R4:ConfigMC(1)->MC.R(IStep)    || R6:ConfigMC(1)->MC.R(IStep)    || R7:ConfigMC(1)->MC.R(IStep);
     R0:ConfigMC(32)->MC.R(ISize)   || R1:ConfigMC(32)->MC.R(ISize)   || R2:ConfigMC(32)->MC.R(ISize)   || R3:ConfigMC(32)->MC.R(ISize)   || R4:ConfigMC(32)->MC.R(ISize)   || R6:ConfigMC(32)->MC.R(ISize)   || R7:ConfigMC(32)->MC.R(ISize);
     R0:ConfigMC(0)->MC.R(Inum)     || R1:ConfigMC(0)->MC.R(Inum)     || R2:ConfigMC(0)->MC.R(Inum)     || R3:ConfigMC(0)->MC.R(Inum)     || R4:ConfigMC(0)->MC.R(Inum)     || R6:ConfigMC(0)->MC.R(Inum)     || R7:ConfigMC(0)->MC.R(Inum);
 
     // read sign bit               || read k1                        || read x_between_0.5_and_8.5                                       || read k0                                                          || read delta
     R0:ConfigMC(32)->MC.R(SStart)  || R1:ConfigMC(224)->MC.R(SStart) || R2:ConfigMC(128)->MC.R(SStart)                                   || R4:ConfigMC(225)->MC.R(SStart)                                   || R7:ConfigMC(192)->MC.R(SStart);
     R0:ConfigMC(1)->MC.R(SStep)    || R1:ConfigMC(2)->MC.R(SStep)    || R2:ConfigMC(1)->MC.R(SStep)                                      || R4:ConfigMC(2)->MC.R(SStep)                                      || R7:ConfigMC(1)->MC.R(SStep);
     R0:ConfigMC(32)->MC.R(SSize)   || R1:ConfigMC(32)->MC.R(SSize)   || R2:ConfigMC(32)->MC.R(SSize)                                     || R4:ConfigMC(32)->MC.R(SSize)                                     || R7:ConfigMC(32)->MC.R(SSize);
     R0:ConfigMC(0)->MC.R(Snum)     || R1:ConfigMC(0)->MC.R(Snum)     || R2:ConfigMC(0)->MC.R(Snum)                                       || R4:ConfigMC(0)->MC.R(Snum)                                       || R7:ConfigMC(0)->MC.R(Snum);
 
     // write port config
     // write x                     || write x^2                      || write y0                       || write x_between_0.5_and_8.5    || write y1                       || write delta                    || write k1 and k0
     R0:ConfigMC(0)->MC.w(IStart)   || R1:ConfigMC(64)->MC.w(IStart)  || R2:ConfigMC(96)->MC.w(IStart)  || R3:ConfigMC(128)->MC.w(IStart) || R4:ConfigMC(160)->MC.w(IStart) || R5:ConfigMC(192)->MC.w(IStart) || R6:ConfigMC(224)->MC.w(IStart);
     R0:ConfigMC(1)->MC.w(IStep)    || R1:ConfigMC(1)->MC.w(IStep)    || R2:ConfigMC(1)->MC.w(IStep)    || R3:ConfigMC(1)->MC.w(IStep)    || R4:ConfigMC(1)->MC.w(IStep)    || R5:ConfigMC(1)->MC.w(IStep)    || R6:ConfigMC(1)->MC.w(IStep);
     R0:ConfigMC(32)->MC.w(ISize)   || R1:ConfigMC(32)->MC.w(ISize)   || R2:ConfigMC(32)->MC.w(ISize)   || R3:ConfigMC(32)->MC.w(ISize)   || R4:ConfigMC(32)->MC.w(ISize)   || R5:ConfigMC(32)->MC.w(ISize)   || R6:ConfigMC(32)->MC.w(ISize);
     R0:ConfigMC(0)->MC.w(Inum)     || R1:ConfigMC(0)->MC.w(Inum)     || R2:ConfigMC(0)->MC.w(Inum)     || R3:ConfigMC(0)->MC.w(Inum)     || R4:ConfigMC(0)->MC.w(Inum)     || R5:ConfigMC(0)->MC.w(Inum)     || R6:ConfigMC(0)->MC.w(Inum);
 
     // write sign bit
     R0:ConfigMC(32)->MC.w(SStart);
     R0:ConfigMC(1)->MC.w(SStep);
     R0:ConfigMC(32)->MC.w(SSize);
     R0:ConfigMC(0)->MC.w(Snum);
 
     BIU0:Wait 0                 || SHU3:Wait 13                         || FMA2:Wait 15                                                                || FMA3:Wait 20                                    || SHU2:Wait 26                                     || Tensor0:Wait 30                              || R3:Wait 30               || R2:Wait 30               || SHU1:Wait 30                    || Tensor2:Wait 34                                  || BIU1:Wait 35                                      || R6:Wait 44               || R7:Wait 44               || Tensor3:Wait 48                           || R1:Wait 50              || FMA1:Wait 54                              || R4:Wait 56                || Tensor1:Wait 60                            || R0:Wait 63              || FMA0:Wait 67                               || SHU0:Wait 71       || BIU3:Wait 74;
 LOOP:
     // load raw data            || abs data                             || x^2                                                                         || write x^2                                       || round(index)                                     || x0 = start + index * step                    || read y0                  || read x^2                 || cast(index)(fp32_i32)           || y1 = a2 + (x^2) * y0                             || read k2, k1                                       || read y1                  || read x^2                 || y2 = a1 + x^2 * y1                        || read x^2                || y3 = a0 + x^2 * y2                        || read x                    || y = x + x * y3                             || read x                  || merge data                                 || y | sign bit       || store result
     //                          || sign bit                             || x[x >= 0x41077800] = 0x41077800                                             || index = -start/step + x * 1/step                                                                                                                                                || read x                                                      || delta = x - x0                                   || read k3, k0                                                                   || read delta               || y0 = k2 + delta * k3                      || read k1                 || y1 = k1 + delta * y0                      || read k0                   || y = k0 + delta * y1                        || read sign bit           || select nan
     //                                                                  || x[x <  0x0319999a] = 0x0319999a                                             || y0 = a3 + (x^2) * a4                                                                                                                                                                                                                                                                                                                                                                                            || passing delta                                                        || passing delta                                                                                                                                   || change nan sign bit
     BIU0:Load(T0)(A++)->SHU3.T0 || SHU3:T0&T6->FMA2.T0 & MC.W0[I++,A++] || FMA2:0+T0*T0(shiftmode0)(FP32)->FMA3.T0                                     || FMA3:0+T0*1(shiftmode0)(FP32)->MC.W1[I++,A++]   || SHU2:Rounding(T0)(fp32)(rInt)->FMA0.T7 & SHU1.T0 || FMAC0:T10+T7*T11(shiftmode0)(FP32)->FMAC2.T7 || R3:MC[I++,A++]->FMAC2.T9 || R2:MC[I++,A++]->FMAC2.T2 || SHU1:Cast(T0)(sat)(fp32_i32)->BIU.T4 || FMAC2:T8+T2*T9(shiftmode0)(FP32)->MC.W4[I++,A++] || BIU1:Disload(T1,T4)(D)->FMAC3.T4 & MC.W6[I++,A++] || R6:MC[I++,A++]->FMAC3.T9 || R7:MC[I++,A++]->FMAC3.T1 || FMAC3:T8+T1*T9(shiftmode0)(FP32)->FMA1.T0 || R1:MC[I++,A++]->FMA1.T1 || FMA1:T8+T1*T0(shiftmode0)(FP32)->FMAC1.T2 || R4:MC[I++,A++]->FMAC1.T8  || FMAC1:T8+T8*T2(shiftmode0)(FP32)->FMA0.T1  || R0:MC[I++,A++]->FMA0.T0 || FALU0:CompSel2(T0,T3,T2,T1)(FP32)->SHU0.T0 || SHU0:T0|T1->BIU.T8 || BIU3:Store(T8,T12)(A++);
                                 || SHU3:T0&T7->MC.W0[S++,A++]           || FALU2:CompSel2(T0,T3,T3,T0)(FP32)->None                                     || FMA3:T10+T7*T11(shiftmode0)(FP32)->SHU2.T0                                                                                                                                      || R2:MC[S++,A++]->FMAC2.T6                                                                                        || BIU1:Disload(T2,T4)(D)->FMAC3.T5 & MC.W6[I++,A++]                             || R7:MC[S++,A++]->FMAC3.T6 || FMAC3:T4+T6*T5(shiftmode0)(FP32)->FMA1.T7 || R1:MC[S++,A++]->FMA1.T5 || FMA1:T5+T6*T7(shiftmode0)(FP32)->FMAC1.T4 || R4:MC[S++,A++]->FMAC1.T11 || FMAC1:T11+T3*T4(shiftmode0)(FP32)->FMA0.T2 || R0:MC[S++,A++]->FMA0.T6 || FALU0:T0>=0(FP32)->None;
                                                                            FALU2:CompSel2(FALU2.Ttmp,T1,FALU2.Ttmp,T1)(FP32)->FMA3.T7 & MC.W3[I++,A++] || FMA3:T8+T0*T9(shiftmode0)(FP32)->MC.W2[I++,A++]                                                                                                                                                                                                || FMAC2:T6-T7*1(shiftmode0)(FP32)->MC.W5[I++,A++]                                                                                                               || FMAC3:0+T6*1(shiftmode0)(FP32)->FMA1.T6                              || FMA1:0+T6*1(shiftmode0)(FP32)->FMAC1.T3                                                                                                         || FALU0:CompSel1(FALU0.Ttmp,T6,0)(FP32)->SHU0.T1                   || MFETCH:if(1 < KI0)JUMP %LOOP (DECKI);
    
     BIU0:Wait 0                 || SHU3:Wait 0                          || FMA2:Wait 0                                                                 || FMA3:Wait 0                                     || SHU2:Wait 0                                      || Tensor0:Wait 0                               || R3:Wait 0                || R2:Wait 0                || SHU1:Wait 0                     || Tensor2:Wait 0                                   || BIU1:Wait 0                                       || R6:Wait 0                || R7:Wait 0                || Tensor3:Wait 0                            || R1:Wait 0               || FMA1:Wait 0                               || R4:Wait 0                 || Tensor1:Wait 0                             || R0:Wait 0               || FMA0:Wait 0                                || SHU0:Wait 0        || BIU3:Wait 0 || MFETCH:MPUWait 0;
 
     MFETCH: Repeat @(4);
 
     MFetch:Jump KI15;