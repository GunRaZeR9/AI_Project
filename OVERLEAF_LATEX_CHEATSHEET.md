# Overleaf LaTeX Cheat Sheet

Acest fisier este gandit ca referinta rapida pentru redactarea unei lucrari in LaTeX pe Overleaf.

## 1. Structura minima a unui document

```latex
\documentclass[12pt,a4paper]{report}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[romanian]{babel}
\usepackage{graphicx}
\usepackage{amsmath, amssymb}
\usepackage{hyperref}

\title{Titlul lucrarii}
\author{Numele Autorului}
\date{\today}

\begin{document}

\maketitle
\tableofcontents

\chapter{Introducere}
Text introductiv.

\section{Context}
Mai mult text.

\end{document}
```

## 2. Sectiuni si numerotare

```latex
\chapter{Capitol}
\section{Sectiune}
\subsection{Subsectiune}
\subsubsection{Subsubsectiune}
```

Pentru documente de tip `article`, de regula nu se foloseste `\chapter`.

## 3. Stiluri de scris

```latex
Text normal

\textbf{Bold}
\textit{Italic}
\underline{Underline}
\emph{Accent logic}

{\large Text mai mare}
{\small Text mai mic}
```

Marimi uzuale:

- `\tiny`
- `\scriptsize`
- `\footnotesize`
- `\small`
- `\normalsize`
- `\large`
- `\Large`
- `\LARGE`
- `\huge`
- `\Huge`

Exemplu:

```latex
{\Large Titlu intermediar}
```

## 4. Paragrafe, alineate, spatii

Paragraf nou: lasa o linie goala intre paragrafe.

```latex
Primul paragraf.

Al doilea paragraf.
```

Linie noua fortata:

```latex
Prima linie \\
A doua linie
```

Spatii utile:

- spatiu normal: se scrie direct
- spatiu fix: `~` de exemplu `Fig.~1`
- spatiu orizontal: `\hspace{1cm}`
- spatiu vertical: `\vspace{0.5cm}`

## 5. Liste cu bullet points si numerotare

Lista cu bullet points:

```latex
\begin{itemize}
    \item Primul punct
    \item Al doilea punct
    \item Al treilea punct
\end{itemize}
```

Lista numerotata:

```latex
\begin{enumerate}
    \item Primul pas
    \item Al doilea pas
    \item Al treilea pas
\end{enumerate}
```

Lista de descriere:

```latex
\begin{description}
    \item[GRU] Gated Recurrent Unit
    \item[LSTM] Long Short-Term Memory
\end{description}
```

## 6. Imagini

In preambul:

```latex
\usepackage{graphicx}
```

Inserare imagine:

```latex
\begin{figure}[h!]
    \centering
    \includegraphics[width=0.7\textwidth]{images/diagrama.png}
    \caption{Diagrama sistemului}
    \label{fig:diagrama}
\end{figure}
```

Optiuni utile:

- `width=0.5\textwidth`
- `height=6cm`
- `scale=0.8`
- `angle=90`

Referire in text:

```latex
Figura~\ref{fig:diagrama} prezinta arhitectura sistemului.
```

Recomandari:

- pastreaza imaginile intr-un folder precum `images/`
- foloseste `\centering` in loc de `center` in interiorul figurii
- pune mereu `\caption` si `\label`
- de regula `\label` se pune dupa `\caption`

## 7. Tabele

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|l|c|r|}
        \hline
        Model & MSE & RMSE \\
        \hline
        RNN  & 0.12 & 0.35 \\
        GRU  & 0.09 & 0.30 \\
        LSTM & 0.08 & 0.28 \\
        \hline
    \end{tabular}
    \caption{Comparatie intre modele}
    \label{tab:modele}
\end{table}
```

Referire in text:

```latex
Tabelul~\ref{tab:modele} sintetizeaza rezultatele.
```

## 8. Formule matematice

Formula in linie:

```latex
$y = ax + b$
```

Formula centrata:

```latex
\[
y = ax + b
\]
```

Formula numerotata:

```latex
\begin{equation}
    y = ax + b
\end{equation}
```

Sistem sau formule aliniate:

```latex
\begin{align}
    h_t &= \mathrm{GRU}(x_t, h_{t-1}) \\
    \hat{x}_{t+1} &= Wh_t + b
\end{align}
```

Pachete utile:

```latex
\usepackage{amsmath, amssymb}
```

## 9. Simboluri utile in matematica

```latex
\alpha, \beta, \gamma, \lambda
\sum, \prod, \int
\frac{a}{b}
\sqrt{x}
\hat{x}, \bar{x}, x_t, x_{t+1}
\leq, \geq, \neq
\in, \subset, \forall, \exists
```

## 10. Citari si bibliografie

Citare simpla in text:

```latex
Conform lucrarii lui Acock~\cite{acock2005missing}, valorile lipsa pot influenta semnificativ analiza.
```

Fisier `.bib` exemplu:

```bibtex
@article{acock2005missing,
  author  = {Acock, Alan C.},
  title   = {Working With Missing Values},
  journal = {Journal of Marriage and Family},
  volume  = {67},
  number  = {4},
  pages   = {1012--1028},
  year    = {2005},
  doi     = {10.1111/j.1741-3737.2005.00191.x}
}
```

In document:

```latex
\bibliographystyle{plain}
\bibliography{references}
```

Sau cu `biblatex`:

```latex
\usepackage[backend=biber,style=apa]{biblatex}
\addbibresource{references.bib}

...

\printbibliography
```

## 11. Linkuri

```latex
\usepackage{hyperref}
```

Exemple:

```latex
\url{https://www.overleaf.com}
\href{https://www.overleaf.com}{Overleaf}
```

## 12. Cod sursa

Varianta simpla:

```latex
\begin{verbatim}
for i in range(10):
    print(i)
\end{verbatim}
```

Varianta mai buna cu pachet dedicat:

```latex
\usepackage{listings}
```

```latex
\begin{lstlisting}[language=Python, caption={Exemplu Python}]
for i in range(10):
    print(i)
\end{lstlisting}
```

## 13. Elemente utile pentru redactarea unei lucrari

Pagina noua:

```latex
\newpage
```

Inceput pagina noua si golire flotanti:

```latex
\clearpage
```

Text fara numerotare la sectiune:

```latex
\section*{Multumiri}
```

Adaugare in cuprins manual:

```latex
\addcontentsline{toc}{section}{Multumiri}
```

## 14. Caractere speciale

Daca vrei sa afisezi caractere rezervate in LaTeX:

```latex
\# \$ \% \& \_ \{ \}
```

Exemplu:

```latex
Costul este 20\% din buget.
```

## 15. Label si ref

Poti pune etichete pentru capitole, figuri, tabele, ecuatii:

```latex
\section{Metodologie}\label{sec:metodologie}
```

Referire:

```latex
In Sectiunea~\ref{sec:metodologie} este descrisa metodologia.
```

## 16. Pachete frecvent utile

```latex
\usepackage{graphicx}   % imagini
\usepackage{amsmath}    % formule
\usepackage{amssymb}    % simboluri matematice
\usepackage{hyperref}   % linkuri
\usepackage{float}      % control pentru [H]
\usepackage{listings}   % cod sursa
\usepackage{xcolor}     % culori
\usepackage{caption}    % captions mai flexibile
\usepackage{subcaption} % subfiguri
```

## 17. Exemplu de subfiguri

```latex
\usepackage{subcaption}
```

```latex
\begin{figure}[h!]
    \centering
    \begin{subfigure}{0.45\textwidth}
        \includegraphics[width=\textwidth]{images/img1.png}
        \caption{Imaginea 1}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.45\textwidth}
        \includegraphics[width=\textwidth]{images/img2.png}
        \caption{Imaginea 2}
    \end{subfigure}
    \caption{Comparatie intre doua imagini}
\end{figure}
```

## 18. Sfaturi practice pentru Overleaf

- organizeaza fisierele in foldere: `chapters/`, `images/`, `bib/`
- foloseste nume simple pentru fisiere, fara spatii
- recompilarea automata te ajuta, dar cand apar erori citeste primul mesaj de eroare, nu ultimul
- daca o figura nu apare unde vrei, incearca `[h!]`, `[t]`, `[b]` sau pachetul `float` cu `[H]`
- pentru bibliografie cu `biblatex`, foloseste `biber`, nu `bibtex`
- dupa modificari la bibliografie, recompilarea poate necesita 1-2 rulari

## 19. Template minim pentru licenta/disertatie

```latex
\documentclass[12pt,a4paper]{report}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[romanian]{babel}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}

\begin{document}

\tableofcontents

\chapter{Introducere}
\section{Incadrare in tematica lucrarii}
Text...

\section{Contributii personale}
Text...

\section{Structura lucrarii}
Text...

\chapter{Aspecte teoretice si tehnologice}
Text...

\chapter{Proiectare}
Text...

\chapter{Implementare}
Text...

\chapter{Rezultate}
Text...

\chapter{Concluzii}
Text...

\end{document}
```

## 20. Cele mai folosite comenzi, pe scurt

- `\textbf{}` pentru bold
- `\textit{}` pentru italic
- `\section{}` pentru sectiuni
- `\begin{itemize} ... \end{itemize}` pentru bullet points
- `\begin{enumerate} ... \end{enumerate}` pentru liste numerotate
- `\includegraphics{}` pentru imagini
- `\caption{}` pentru descriere figura/tabel
- `\label{}` si `\ref{}` pentru referinte interne
- `$...$` sau `\[ ... \]` pentru formule
- `\cite{}` pentru citari
- `\bibliography{}` sau `\printbibliography` pentru bibliografie
