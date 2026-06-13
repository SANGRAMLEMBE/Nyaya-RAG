# Corpus probe report

Use this report to design the extraction and section-tree parser.

| doc | pages | chars/page | text pages | verdict |
|---|---|---|---|---|
| INDIA CODE | 5 | 2526 | 100.0% | **digital** |
| bns_2023 | 112 | 3571 | 100.0% | **digital** |
| bnss_2023 | 279 | 2923 | 100.0% | **digital** |
| bsa_2023 | 54 | 3118 | 100.0% | **digital** |
| central acts | 26 | 4793 | 92.3% | **digital** |
| constitution | 799 | 2428 | 99.5% | **digital** |
| constitution of india | 402 | 2083 | 100.0% | **digital** |
| contract_1872 | 53 | 3344 | 100.0% | **digital** |
| cpa_2019 | 39 | 3448 | 100.0% | **digital** |
| crpc_1973 | 263 | 3330 | 100.0% | **digital** |
| hma_1955 | 17 | 2835 | 94.1% | **digital** |
| iea_1872 | 91 | 2670 | 100.0% | **digital** |
| ipc_1860 | 227 | 2742 | 100.0% | **digital** |
| it_act_2000 | 41 | 3266 | 100.0% | **digital** |
| lsa_1987 | 19 | 3076 | 100.0% | **digital** |
| mva_1988 | 175 | 2543 | 70.9% | **mixed** |
| ni_act_1881 | 32 | 3208 | 100.0% | **digital** |
| posh_2013 | 14 | 2836 | 100.0% | **digital** |
| pwdva_2005 | 12 | 3153 | 100.0% | **digital** |
| rti_2005 | 25 | 2830 | 100.0% | **digital** |

## INDIA CODE
- header candidates: none
- footer candidates: ['https://www.indiacode.nic.in/bitstream/#/#/#/A#-#.pdf#search=Indian%#Contract%#Act', 'https://www.indiacode.nic.in/bitstream/#/#/#/#.pdf#search=Legal%#Services%#Authorities']
- body sample: `Indian Evidence Act https://www.indiacode.nic.in/bitstream/123456789/2314/1/A1888-4.pdf#search=Indian%20Evidence%20Act https://www.indiacode.nic.in/bitstream/123456789/1743/1/A1975-20.pdf#search=Indian%20Evidence%20Act https://www.indiacode.nic.in/bitstream/123456789/2309/1/A1886-6.pdf#search=Indian%20Evidence%20Act https://www.indiacode.nic.in/bitstream/123456789/2157/3/A2016-11.pdf#search=Indian`

## bns_2023
- header candidates: ['#']
- footer candidates: none
- body sample: `2 LIST OF ABBREVIATIONS USED G.S.R. . . . . . for General Statutory Rules. S.O. . . . . . ,, Statutory Order. Notifn. . . . . . ,, Notification. `

## bnss_2023
- header candidates: ['#']
- footer candidates: none
- body sample: `2 CHAPTER IV POWERS OF SUPERIOR OFFICERS OF POLICE AND AID TO THE MAGISTRATES AND THE POLICE SECTIONS 30. Powers of superior officers of police. 31. Public when to assist Magistrates and police. 32. Aid to person, other than police officer, executing warrant. 33. Public to give information of certain offences. 34. Duty of officers employed in connection with affairs of a village to make certain re`

## bsa_2023
- header candidates: ['#']
- footer candidates: none
- body sample: `2 LIST OF ABBREVIATIONS USED G.S.R. . . . . . for General Statutory Rules. S.O. . . . . . ,, Statutory Order. Notifn. . . . . . ,, Notification. `

## central acts
- header candidates: ['#']
- footer candidates: none
- body sample: ` केन्‍द द्रीय अधिधियम A Absorbed Areas (Laws) Act, 1954 आमेधलत क्षेत्र (धवधि) अधिधियम, 1954 (1954 का 20) Academy of Scientific and Innovative Research Act, 2011 वैज्ञाधिक और प्रवर्तणत अिुसंिाि अकादमी अधिधियम, 2011 (2012 का 13) Acquired Territories (Merger) Act, 1960 अर्जणत राज् यक्षेत्र (धवलयि) अधिधियम, 1960 (1960 का 64) Acquisition of Certain Area at Ayodhya Act, 1993 अयोध्या में कधतपय क्षेत्र अज`

## constitution
- header candidates: ['#', '¨sÁgÀvÀzÀ ¸ÀA«zsÁ£À', 'The Constitution of India']
- footer candidates: none
- body sample: ` ¨sÁgÀvÀzÀ ¸ÀA«zsÁ£À [11£ÉÃ £ÀªÉA§gï, 2025 gÀAzÀÄ EzÀÝAvÉ] THE CONSTITUTION OF INDIA [As on 11th November, 2025] ¨sÁgÀvÀ ¸ÀPÁðgÀ PÁ£ÀÆ£ÀÄ ªÀÄvÀÄÛ £ÁåAiÀÄ ªÀÄAvÁæ®AiÀÄ «zsÁ¬ÄÃ E¯ÁSÉ GOVERNMENT OF INDIA Ministry of Law and Justice Legislative Department 2025 `

## constitution of india
- header candidates: none
- footer candidates: none
- body sample: ` PREFACE This is the sixth pocket size edition of the Constitution of India in the diglot form. In this edition, the text of the Constitution of India has been brought up-to-date by incorporating therein all the amendments up to the Constitution (One Hundred and Sixth Amendment) Act, 2023. The foot notes below the text indicate the Constitution Amendment Acts by which such amendments have been mad`

## contract_1872
- header candidates: ['#']
- footer candidates: none
- body sample: `2 SECTIONS 23. What considerations and objects are lawful, and what not. Void agreements 24. Agreement void, if considerations and objects unlawful in part. 25. Agreement without consideration, void, unless it is in writing and registered, or is a promise to compensate for something done, or is a promise to pay a debt barred by limitation law. 26. Agreement in restraint of marriage, void. 27. Agre`

## cpa_2019
- header candidates: ['#']
- footer candidates: none
- body sample: `2 SECTIONS 30. Salaries, allowances and other terms and conditions of service of President and members of District Commission. 31. Transitional provision. 32. Vacancy in office of member of District Commission. 33. Officers and other employees of District Commission. 34. Jurisdiction of District Commission. 35. Manner in which complaint shall be made. 36. Proceedings before District Commission. 37`

## crpc_1973
- header candidates: ['#']
- footer candidates: none
- body sample: `2 SECTIONS 28. Sentences which High Courts and Sessions Judges may pass. 29. Sentences which Magistrates may pass. 30. Sentence of imprisonment in default of fine. 31. Sentence in cases of conviction of several offences at one trial. 32. Mode of conferring powers. 33. Powers of officers appointed. 34. Withdrawal of powers. 35. Powers of Judges and Magistrates exercisable by their successors-in-off`

## hma_1955
- header candidates: ['#']
- footer candidates: none
- body sample: `2 LIST OF AMENDING ACTS 1. The Hindu Marriage (Amendment) Act, 1956 (73 of 1956). 2. The Hindu Marriage (Amendment) Act, 1964 (44 of 1964). 3. The Marriage Laws (Amendment) Act, 1976 (68 of 1976). 4. The Child Marriage Restraint (Amendment) Act, 1978 (2 of 1978). 5. The Marriage Laws (Amendment) Act, 1999 (39 of 1999). 6. The Marriage Laws (Amendment) Act, 2001 (49 of 2001). 7. The Marriage Laws (`

## iea_1872
- header candidates: ['---------------------------------------------------------------------', '#']
- footer candidates: none
- body sample: ` --------------------------------------------------------------------- --------------------------------------------------------------------- 5*[or the Air Force Act] (7 Geo. 5, c. 51.) but not to affidavits 6* presented to any Court or officer, nor to proceedings before an arbitrator; Commencement of Act. Commencement of Act.-And it shall come into force on the first day of September, 1872. 2. Rep`

## ipc_1860
- header candidates: ['----------------------------------------------------------------------']
- footer candidates: none
- body sample: ` ---------------------------------------------------------------------- 3. Punishment of offences committed beyond, but which by law may be tried within, India.--Any person liable, by any 7*[Indian law], to be tried for an offence committed beyond 5*[India] shall be dealt with according to the provisions of this Code for any act committed beyond 5*[India] in the same manner as if such act had been`

## it_act_2000
- header candidates: ['#']
- footer candidates: none
- body sample: `2 SECTIONS 27. Power to delegate. 28. Power to investigate contraventions. 29. Access to computers and data. 30. Certifying Authority to follow certain procedures. 31. Certifying Authority to ensure compliance of the Act, etc. 32. Display of licence. 33. Surrender of licence. 34. Disclosure. CHAPTER VII ELECTRONIC SIGNATURE CERTIFICATES 35. Certifying authority to issue electronic signature Certif`

## lsa_1987
- header candidates: ['#']
- footer candidates: none
- body sample: `2 SECTIONS 15. National Legal Aid Fund. 16. State Legal Aid Fund. 17. District Legal Aid Fund. 18. Accounts and audit. CHAPTER VI LOKADALATS 19. Organisation of Lok Adalats. 20. Cognizance of cases by Lok Adalats. 21. Award of Lok Adalat. 22. Powers of Lok Adalat or Permanent Lok Adalat. CHAPTER VIA PRE-LITIGATION CONCILIATION AND SETTLEMENT 22A. Definitions. 22B. Establishment of Permanent Lok Ad`

## mva_1988
- header candidates: ['#']
- footer candidates: none
- body sample: `2 LIST OF AMENDING ACTS 1. The Motor Vehicles (Amendment) Act, 1994 (54 of 1994). 2. The Motor Vehicles (Amendment) Act, 2000 (27 of 2000). 3. The Motor Vehicles (Amendment) Act, 2001 (39 of 2001). 4. The Motor Vehicles (Amendment) Act, 2015 (3 of 2015). 5. The Motor Vehicles (Amendment) Act, 2019 (32 of 2019). 6. The Jan Vishwas (Amendment of Provisions) Act, 2023 (18 of 2023). _______ LIST OF AB`

## ni_act_1881
- header candidates: ['#']
- footer candidates: none
- body sample: `2 SECTIONS 16. Indorsement “in blank” and “in full”. “Indorsee.” 17. Ambiguous instruments. 18. Where amount is stated differently in figures and words. 19. Instruments payable on demand. 20. Inchoate stamped instruments. 21. “At sight” —“On presentment.” “After sight.” 22. “Maturity.” Days of grace. 23. Calculating maturity of bill or note payable so many months after date or sight. 24. Calculati`

## posh_2013
- header candidates: ['#']
- footer candidates: none
- body sample: `2 CHAPTER VI DUTIES OF EMPLOYER SECTIONS 19. Duties of employer. CHAPTER VII DUTIES AND POWERS OF DISTRICT OFFICER 20. Duties and powers of District Officer. CHAPTER VIII MISCELLANEOUS 21. Committee to submit annual report. 22. Employer to include information in annual report. 23. Appropriate Government to monitor implementation and maintain data. 24. Appropriate Government to take measures to pub`

## pwdva_2005
- header candidates: ['#']
- footer candidates: none
- body sample: `2 SECTIONS 29. Appeal. CHAPTER V MISCELLANEOUS 30. Protection Officers and members of service providers to be public servants. 31. Penalty for breach of protection order by respondent. 32. Cognizance and proof. 33. Penalty for not discharging duty by Protection Officer. 34. Cognizance of offence committed by Protection Officer. 35. Protection of action taken in good faith. 36. Act not in derogatio`

## rti_2005
- header candidates: ['#']
- footer candidates: none
- body sample: `2 LIST OF AMENDING ACTS 1. The Right to Information (Amendment) Act, 2019 (24 of 2019). 2. The Jammu and Kashmir Reorganisation Act, 2019 (34 of 2019). 3. The Digital Personal Data Protection Act, 2023 (22 of 2023). _______ LIST OF ABBREVIATIONS USED Cl., cls. . . . . . for Clause, clauses. Ins. . . . . . ,, Inserted. Notifn. . . . . . ,, Notification. S., ss. . . . . . ,, Section, sections. Sch. `
