% !TEX root = ornl-template-example.tex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: ornltm.cls
% Authors: Seth R Johnson, Sam Crawford,* and John Batson
% *Corresponding author, crawfordst.@ornl.gov
% Date: Monday April 29 13:41:46 2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{ornltm}[2018/10/13 ORNL Technical Memorandum class]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ProcessOptions\relax
\LoadClass[11pt,letterpaper]{article}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STYLING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{wallpaper} % background on cover page
\RequirePackage{adjustbox} % used to keep author block within front cover
\RequirePackage{fancyhdr} % headers and footers
\RequirePackage{emptypage} % hide footers on blank pages
\RequirePackage[pdfusetitle]{hyperref} % url and links in PDF
\RequirePackage{acro} % Add support for abbreviations
\RequirePackage{array} % Add support for tabulars
\RequirePackage{tocloft} % table of contents styling
\RequirePackage{color} % colored fonts and boxes
\RequirePackage{framed} % framed notice boxes
\RequirePackage{seqsplit} % for splitting unruly character strings
\RequirePackage{etoolbox} % Patch for all caps first- and second-order headings in TOC

% Load document style
\RequirePackage{ornltm-style}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARIABLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Division or program name
\newif\if@divisiondefined
\@divisiondefinedfalse
\def\@division{DIVISION NAME UNDEFINED%
\PackageWarning{ornltm}{No division name has been defined.}}
\newcommand{\division}[1]{\def\@division{#1}\@divisiondefinedtrue}

% Use the \reportnum command to define a report number.
\newif\if@reportnumdefined
\@reportnumdefinedfalse
\def\@reportnum{ORNL/XX-XXXX/XXX%
\PackageWarning{ornltm}{No report number has been defined.}}
\newcommand{\reportnum}[1]{\def\@reportnum{#1}\@reportnumdefinedtrue}

% Use the \sponsornum command to define a sponsor number.
\newif\if@sponsornumdefined
\@sponsornumdefinedfalse
\def\@sponsornum{%
}
\newcommand{\sponsornum}[1]{\def\@sponsornum{#1}\@sponsornumdefinedtrue}

\newif\if@reportdraft
\@reportdraftfalse
\newcommand{\reportdraft}{\@reportdrafttrue}

\def\@affiliation{\relax}
\newcommand{\affiliation}[1]{\def\@affiliation{#1}}
\newcommand{\affilnum}[1]{\texorpdfstring{\textsuperscript{#1}}{}}

% Adding sponsor logo definitions.
\newif\if@SponsorLogoOnedefined
\@SponsorLogoOnedefinedfalse
\newcommand{\SponsorLogoOne}[1]{\def\@SponsorLogoOne{#1}\@SponsorLogoOnedefinedtrue}

\newif\if@SponsorLogoTwodefined
\@SponsorLogoTwodefinedfalse
\newcommand{\SponsorLogoTwo}[1]{\def\@SponsorLogoTwo{#1}\@SponsorLogoTwodefinedtrue}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Added for CRADA (remains blank if CRADA is undefined)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Uncomment the \crada{...} command in the document and add number.
\newif\if@cradadefined
\@cradadefinedfalse
\newcommand{\crada}[1]{\def\@crada{#1}\@cradadefinedtrue}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Added for CUI (remains blank if CUI is undefined)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Uncomment the \cui{...} command in the document if CUI.
\newif\if@cuidefined
\@cuidefinedfalse
\newcommand{\cui}[1]{\def\@cui{#1}\@cuidefinedtrue}

\newif\if@cuiDAdefined
\@cuiDAdefinedfalse
\def\@cuiDA{CUI DA UNDEFINED%
\PackageWarning{ornltm}{No CUI designated authority has been defined.}}
\newcommand{\cuiDA}[1]{\def\@cuiDA{#1}\@cuiDAdefinedtrue}

\newif\if@cuiPhdefined
\@cuiPhdefinedfalse
\def\@cuiPh{CUI DA PHONE UNDEFINED%
\PackageWarning{ornltm}{No CUI designated authority phone number has been defined.}}
\newcommand{\cuiPh}[1]{\def\@cuiPh{#1}\@cuiPhdefinedtrue}

% Creating \printcui command to simplify CUI implementation
\newcommand{\printcui}{
\begin{center}
\if@cuidefined{\large\textbsf{\par\@cui}}
\else\relax
\fi%
\end{center}
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use the \restrict command to add qualifications to the document's restriction.
% (e.g., business sensitive).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newif\if@restrictdefined
\@restrictdefinedfalse
\newcommand{\restrict}[1]{\def\@restrict{#1}\@restrictdefinedtrue}

% Creating \printrestrict command to simplify restricted footer
\newcommand{\printrestrict}{
  \if@cuidefined % CUI supersedes restrict, so do nothing if CUI.
  \else%
  \if@restrictdefined{\centering\large\textbsf{\@restrict}\par}
  \else\relax%
  \fi%
  \fi
  }  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define ORNL title photo with default
\def\@ornltmtitlephoto{luggage/ornltm-titlephoto.jpg}
\newcommand{\reportphoto}[1]{\def\@ornltmtitlephoto{#1}}

% Define uppercase names
\newcommand\ornltmabstractname{ABSTRACT}
\newcommand\ornltmlofname{LIST OF FIGURES}
\newcommand\ornltmlotname{LIST OF TABLES}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COVER PAGE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Bold sans-serif
\newcommand{\textbsf}[1]{{\bfseries\sffamily #1}}
% Bold italics
\newcommand{\textbi}[1]{{\bfseries\itshape #1}}
% ORNL official green color
\definecolor{ornlgreen}{RGB}{0,121,52}
% Green can render as this color (???)
\definecolor{ornldarkgreen}{RGB}{52,118,60} % from included ORNL logo

% Footer for cover page
\fancypagestyle{coverpage}
{
  \renewcommand\headrulewidth{0pt}
  \fancyhf{}
}

\newcommand{\@restrictfootwarning}[1]{%
  \vspace{-2\baselineskip}\par
  \makebox[\textwidth][c]{\textbsf{#1}}\par
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Header and footer for fancy pages
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pagestyle{fancy}

\fancyhead{}
% Adding CUI to header if defined.
\chead{%
  \if@cuidefined%
    \makebox[\textwidth][c]{\textbsf{\@cui}}
  \fi%
}
% Adding CUI or restricted to footer if defined. CUI supersedes restrict.
\fancyfoot{}
\cfoot{%
  \if@cuidefined%
    \@restrictfootwarning{\@cui}%
  \else\if@restrictdefined%
    \@restrictfootwarning{\@restrict}%
  \fi\fi%
  \thepage{}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Header and footer for plain pages (i.e., most of the document)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\fancypagestyle{plain}

\fancyhead{}
% Adding CUI to header if defined.
\chead{%
 \if@cuidefined%
   \makebox[\textwidth][c]{\textbsf{\@cui}}%
 \fi%
}
% Adding CUI or restricted to footer if defined.
\cfoot{%
  \if@cuidefined%
    \@restrictfootwarning{\@cui}%
  \else\if@restrictdefined%
    \@restrictfootwarning{\@restrict}%
  \fi\fi%
  \thepage{}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BEGIN COVER OPTIONS FOR DIFFERENT REPORT TYPES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\headlinecolor}{\normalcolor}
\RequirePackage{xcolor}
\definecolor{slcolor}{HTML}{882B21}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TECHNICAL MEMO (TM) OPTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\DeclareOption{tm}{%
%
\def\makecoverpage{%
\newgeometry{top=0.5in,bottom=0.5in,right=.5in,left=2in}

\begin{titlepage}
\thispagestyle{coverpage}
\ThisULCornerWallPaper{1.0}{luggage/ornltm-sidebar.jpg}%

% Added for CUI (remains blank if CUI is undefined)
\printcui

%% REPORT NUMBER %%
\vspace*{0.5\baselineskip plus 1\baselineskip}
\begin{minipage}{\textwidth}
\raggedleft
\large\textbsf{\@reportnum}
\end{minipage}%
\vspace*{0.5\baselineskip plus 0.5fill}

%% TITLE %%
\begin{minipage}{\textwidth}%
  \Huge\textbsf{%
    \@title
  }
\end{minipage}%

\vspace*{1\baselineskip plus 1\baselineskip minus 1\baselineskip}

%%% PHOTO %%%
\includegraphics[width=6.0in, height=4.0in]{luggage/ornltm-titlephoto.jpg}

%%% NOTICES %%%
% If restrict and draft, show restrict + draft
% If restrict and not draft, show restrict
% If not restrict and draft, show draft
% If not restrict and not draft, show ``approved for public release"
\vspace*{0pt \@plus 1fill}

%%% AUTHOR AND DATE %%%
\begin{minipage}[c]{3in}%
\def\and{\\}% Newline between authors
\large\sffamily%
\setlength\tabcolsep{0pt}%
\maxsizebox*{!}{2in}{% Restrict size of author box.
\begin{tabular}[t]{l}%
\@author%
\end{tabular}
}
\vspace*{1\baselineskip \@plus 1\baselineskip \@minus 1\baselineskip}

\renewcommand{\today}{\ifcase \month \or January\or February\or March\or %
April\or May \or June\or July\or August\or September\or October\or November\or %
December\fi~\number \year}

\large\textbsf{\@date}
\end{minipage}%
%
\begin{minipage}[c]{3in}%


% Check if document is a draft to determine boxed notices.
\if@reportdraft{
\begin{center}
  \textbsf{\large Draft. Document has not been reviewed and approved for public release.}%
\end{center}
\vspace{1em}
}
\fi

% Configure notices for front cover.
\if@cuidefined%
\relax % No CUI notice needed, so do nothing.
\else % Otherwise, check for restrict.
\if@restrictdefined{% If restrict is defined in document, then print box.
\begin{center}
\printrestrict%
\end{center}
\vspace{1em}
}%
\fi
\fi

\if@cradadefined{% CRADA is independent. Print box if defined.
\begin{center}
\textbsf{\large\par CRADA final report for \\
\@crada}%
\end{center}
\vspace{1em}
}%
\fi

% Ensure restricted and CUI documents are *not* marked for public release
% if the document is *not* marked as draft.
\if@cuidefined{%
\begin{center}
\textbsf{\large Controlled by: \@cuiDA, \@cuiPh}%
\end{center}
}\else%
\if@restrictdefined{%
}\else%
\if@reportdraft\relax\else{%
\begin{center}
\textbsf{\large% If not draft, CUI, or otherwise restricted, then unlimited distro.
  Approved for public release. \\
  Distribution is unlimited.}%
\end{center}
}\fi%
\fi%
\fi

\end{minipage}

\vspace*{0pt \@plus 1fill}

\noindent\includegraphics[width=2.08in]{luggage/ornltm-logo.jpg}\\[-.15in]
\noindent\textcolor{ornlgreen}{\rule{6in}{.5pt}}\\
\noindent{\footnotesize\sffamily ORNL IS MANAGED BY UT-BATTELLE LLC FOR THE US~DEPARTMENT OF ENERGY}

% Added for CUI (remains blank if \cui is undefined)
\printcui
% Added for restrict (remains blank if \restrict is undefined)
\printrestrict

\end{titlepage}
\restoregeometry
}%
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SPONSOR REPORT (SPR) OPTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\DeclareOption{spr}{%

\def\makecoverpage{%
\newgeometry{top=0.25in,bottom=0.25in,right=2in,left=.5in}

\begin{titlepage}
\thispagestyle{coverpage}
\ThisURCornerWallPaper{1.0}{luggage/ornlspr-sidebar.jpg}%

% Added for CUI (remains blank if CUI is undefined)
\printcui

%% REPORT NUMBER %%
\vspace*{0.5\baselineskip plus 1\baselineskip}
\begin{minipage}{\textwidth}
\flushleft
\large\textbsf{\@reportnum} \\
\large\textbsf{\@sponsornum}
\end{minipage}%
\vspace*{0.5\baselineskip plus 0.5fill}
%

%% TITLE %%
\begin{minipage}{\textwidth}%
  \Huge\textbsf{%
    \@title
  }
\end{minipage}%

\vspace*{1\baselineskip plus 1\baselineskip minus 1\baselineskip}

%%% PHOTO %%%
\includegraphics[width=6.0in, height=4.0in]{luggage/ornlspr-titlephoto.jpg}

\vspace{\stretch{1}}

%%% NOTICES %%%
% If restrict and draft, show restrict + draft
% If restrict and not draft, show restrict
% If not restrict and draft, show draft
% If not restrict and not draft, show ``approved for public release"

\begin{minipage}[c]{3in}%
% Check if document is a draft to determine boxed notices.
\if@reportdraft{
\begin{center}
  \textbsf{\large Draft. Document has not been reviewed and approved for public release.}%
\end{center}
\vspace{1em}
}
\fi

% Configure notices for front cover.
\if@cuidefined%
\relax % No CUI notice needed, so do nothing.
\else % Otherwise, check for restrict.
\if@restrictdefined{% If restrict is defined in document, then print box.
\begin{center}
\printrestrict%
\end{center}
\vspace{1em}
}%
\fi
\fi

\if@cradadefined{% CRADA is independent. Print box if defined.
\begin{center}
\textbsf{\large\par CRADA final report for \\
\@crada \vspace{1em}}%
\end{center}
}%
\fi


% Ensure restricted and CUI documents are *not* marked for public release
% if the document is *not* marked as draft.
\if@cuidefined{%
\begin{center}
\textbsf{\large Controlled by: \@cuiDA, \@cuiPh}%
\end{center}
}\else%
\if@restrictdefined{%
}\else%
\if@reportdraft\relax\else{%
\begin{center}
\textbsf{\large% If not draft, CUI, or otherwise restricted, then unlimited distro.
  Approved for public release. \\
  Distribution is unlimited.}%
\end{center}
}\fi%
\fi%
\fi

\end{minipage}
%
\hfill{}\hspace{0.05\textwidth}
%%% AUTHOR AND DATE %%%
\begin{minipage}[c]{3in}%
\def\and{\\}% Newline between authors
\large\sffamily%
\setlength\tabcolsep{0pt}%
\maxsizebox*{!}{2.5in}{% Restricting size of author box.
\begin{tabular}[t]{l}%
\@author%
\end{tabular}
}

\vspace*{1\baselineskip \@plus 1\baselineskip \@minus 1\baselineskip}

\renewcommand{\today}{\ifcase \month \or January\or February\or March\or %
April\or May \or June\or July\or August\or September\or October\or November\or %
December\fi~\number \year}

\large\textbsf{\@date}
\end{minipage}%


\vspace*{0pt \@plus 1fill}

\begin{minipage}[b]{3in}%
\footnotesize\textsf\@affiliation
\vspace{1ex}\par
\end{minipage}%  

\noindent\includegraphics[width=2.08in]{luggage/ornlspr-logo.jpg}\hspace{0.05\textwidth}
\if@SponsorLogoOnedefined{
\noindent\includegraphics[width=1.45in]{\@SponsorLogoOne}\hspace{0.05\textwidth}
}\else\fi
\if@SponsorLogoTwodefined{
\noindent\includegraphics[width=1.45in]{\@SponsorLogoTwo}
}\else\fi
\\[-.15in]
\noindent\textcolor{ornlgreen}{\rule{6in}{.5pt}}\\
\noindent{\footnotesize\sffamily ORNL IS MANAGED BY UT--BATTELLE LLC FOR THE US~DEPARTMENT OF ENERGY}

% Added for CUI (remains blank if \cui is undefined)
\printcui
% Added for restrict (remains blank if \restrict is undefined)
\printrestrict

\end{titlepage}
\restoregeometry
}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ORNL LETTER (LTR) OPTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\DeclareOption{ltr}{%
%
\def\makecoverpage{%
\newgeometry{top=0.35in,bottom=0.35in,right=0.5in,left=1.25in}

\begin{titlepage}
\thispagestyle{coverpage}
\ThisULCornerWallPaper{1.0}{luggage/ltr-background.png}%

% Added for CUI (remains blank if CUI is undefined)
\printcui

%% REPORT NUMBER %%
\vspace*{0.5\baselineskip plus 1\baselineskip}
\begin{minipage}{\textwidth}
\raggedleft
\large\textbsf{\@reportnum}
\end{minipage}%

\vspace*{0.5\baselineskip plus 0.5fill}

%% TITLE %%
\begin{minipage}{\textwidth}%
  \Huge\textbsf{%
    \@title
  }
\end{minipage}%

\vspace*{1\baselineskip plus 1\baselineskip minus 1\baselineskip}

%%% GRADIENT LINE %%%
\hspace*{-1.25in}\includegraphics[width=8.5in, height=0.13in]{luggage/ltr-gradient-bar.png}%
\vspace*{3.25in}

%%% NOTICES %%%
% If restrict and draft, show restrict + draft
% If restrict and not draft, show restrict
% If not restrict and draft, show draft
% If not restrict and not draft, show ``approved for public release"

%%% AUTHOR AND DATE %%%
\begin{minipage}[c]{3in}%
  \def\and{\\}% Newline between authors
  \large\sffamily%
  \setlength\tabcolsep{0pt}%
  \maxsizebox*{!}{2.5in}{%
  \begin{tabular}[t]{l}%
  \@author%
  \end{tabular}
  }

\vspace*{1\baselineskip \@plus 1\baselineskip \@minus 1\baselineskip}

\renewcommand{\today}{\ifcase \month \or January\or February\or March\or %
April\or May \or June\or July\or August\or September\or October\or November\or %
December\fi~\number \year}

\large\textbsf{\@date}
\end{minipage}%
%
\begin{minipage}[c]{3in}%

% Check if document is a draft to determine boxed notices.
\if@reportdraft{
\begin{center}
  \textbsf{\large Draft. Document has not been reviewed and approved for public release.}%
\end{center}
\vspace{1em}
}
\fi

% Configure notices for front cover.
\if@cuidefined{% No CUI notice needed, so do nothing.
}\else % Otherwise, check for restrict.
\if@restrictdefined{% If restrict is defined in document, then print box.
\begin{center}
\printrestrict%
\end{center}
\vspace{1em}
}
\fi
\fi

\if@cradadefined{% CRADA is independent. Print box if defined.
\begin{center}
\textbsf{\large\par CRADA final report for \\
\@crada \vspace{1em}}%
\end{center}
}
\fi

% Ensure restricted and CUI documents are *not* marked for public release
% if the document is *not* marked as draft.
\if@cuidefined{%
\begin{center}
\textbsf{\large Controlled by: \@cuiDA, \@cuiPh}%
\end{center}
}\else%
\if@restrictdefined{%
}\else%
\if@reportdraft\relax\else{%
\begin{center}
\textbsf{\large% If not draft, CUI, or otherwise restricted, then unlimited distro.
  Approved for public release. \\
  Distribution is unlimited.}%
\end{center}
}\fi%
\fi%
\fi

\end{minipage}

\vspace*{0pt \@plus 1fill}

\begin{minipage}{\textwidth}%
\hspace*{-0.25in}\textcolor{white}{{\footnotesize\sffamily ORNL IS MANAGED BY UT-BATTELLE LLC FOR THE US~DEPARTMENT OF ENERGY}}
\vspace{3em}  
%
\if@restrictdefined{
\if@cuidefined{ % If CUI defined, then do nothing. CUI printed below.
}\else % Otherwise, use the following spacing.
	\vspace{-1.5em}
	\textcolor{white}{\printrestrict}%
\fi
}\else
\fi  	
%
\if@cuidefined{
	\vspace{-1.5em}
	\textcolor{white}{\printcui} 
}\else
%
\fi  	
\end{minipage}%
\end{titlepage}
\restoregeometry
}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\ExecuteOptions{tm} % Defaults to TM if no option is set.
\ProcessOptions\relax

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END COVER OPTIONS FOR DIFFERENT REPORT TYPES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OSTI NOTICE PAGE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Email URL command
\newcommand{\email}[1]{\href{mailto:#1}{#1}}

% Create page
\def\makeostipage{%
\begin{titlepage}

% Added for CUI (remains blank if CUI is undefined)
\printcui


\vspace*{.5in}

  \centering\sffamily\small
  \fbox{\parbox{5.61in}{%
    \begin{center}
      \textbf{DOCUMENT AVAILABILITY}
    \end{center}

\textbf{\textit{Online Access:}} US Department of Energy (DOE) reports produced after 1991 and a growing number of pre-1991 documents are available free via \href{https://www.osti.gov/}{https://www.osti.gov/}.  \\

The public may also search the National Technical Information Service's \href{https://ntrl.ntis.gov/NTRL/}{National Technical Reports Library (NTRL)} for reports not available in digital format. \\

DOE and DOE contractors should contact DOE's Office of Scientific and Technical Information (OSTI) for reports not currently available in digital format: \\

\begin{quote}
US Department of Energy \\
Office of Scientific and Technical Information \\
PO Box 62 \\
Oak Ridge, TN 37831-0062 \\
\textbf{\textit{Telephone:}} (865) 576-8401 \\
\textbf{\textit{Fax:}} (865) 576-5728 \\
\textbf{\textit{Email:}}  \href{mailto:reports@osti.gov}{reports@osti.gov} \\
\textbf{\textit{Website:}} \href{https://www.osti.gov/}{https://www.osti.gov/} \\
\end{quote}

}}

\vspace{\fill}

\fbox{\parbox{4in}{
This report was prepared as an account of work sponsored by an agency of the
United States Government. Neither the United States Government nor any agency
thereof, nor any of their employees, makes any warranty, express or implied, or
assumes any legal liability or responsibility for the accuracy, completeness, or
usefulness of any information, apparatus, product, or process disclosed, or
represents that its use would not infringe privately owned rights. Reference
herein to any specific commercial product, process, or service by trade name,
trademark, manufacturer, or otherwise, does not necessarily constitute or imply
its endorsement, recommendation, or favoring by the United States Government or
any agency thereof. The views and opinions of authors expressed herein do not
necessarily state or reflect those of the United States Government or any agency
thereof.}}

\vspace*{.5in}

% Added for CUI (remains blank if CUI is undefined)
\printcui
% Added for restrict (remains blank if \restrict is undefined)
\printrestrict

\end{titlepage}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TITLE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand\maketitlepage{%
\begin{titlepage}
\setcounter{page}{2}%

% Added for CUI (remains blank if CUI is undefined)
\printcui

%%% REPORT NUMBER %%%
\raggedleft{\large\textbf{\@reportnum}}
\vspace{1in}

%%% DIVISION %%%
\centering\@division
\vspace{\stretch{1}}

%%% TITLE %%%
{\Large\textbf{\MakeUppercase{\@title}}}
\vspace{\stretch{1}}%

%%% AUTHOR %%%
\maxsizebox*{!}{3.75in}{% Restrict size of author box.
\begin{tabular}[t]{c}%
 \@author%
\end{tabular}%
}

\vspace{\stretch{.5}}

% Date Published
\@date

\vspace{\stretch{.5}}

Prepared by \\
OAK RIDGE NATIONAL LABORATORY \\
Oak Ridge, TN 37831 \\
managed by \\
UT-BATTELLE LLC \\
for the \\
US DEPARTMENT OF ENERGY \\
under contract DE-AC05-00OR22725

\vspace{\stretch{.5}}

% Added for CUI (remains blank if CUI is undefined)
\printcui
% Added for restrict (remains blank if \restrict is undefined)
\printrestrict

\end{titlepage}%
} % end \maketitlepage

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FRONT MATTER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newif\if@mainmatter
\@mainmattertrue
\def\@cleariffrontmatter{\if@mainmatter\relax\else{\clearpage}\fi}
\def\@cleardbliffrontmatter{\if@mainmatter\relax\else{\cleardoublepage}\fi}

\newcommand\frontmatter{%
  \@mainmatterfalse%
  \renewcommand\and{\\}
  \renewcommand\thepage{Front cover}%
  \makecoverpage%
  \pagenumbering{roman}%
  \setcounter{page}{1}%
  \makeostipage%
  \maketitlepage%
  \cleardoublepage
  \setcounter{page}{3}%
}

% Abstract gets a new page if it's in the front matter; otherwise it's just at
% the top of the page.
\renewenvironment{abstract}{%
\if@mainmatter\relax\else
  \cleardoublepage
	\pagenumbering{arabic}
	\setcounter{page}{1}
\fi
	\addcontentsline{toc}{section}{\ornltmabstractname}
	\phantomsection
  \section*{\ornltmabstractname}%
}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TABLE OF CONTENTS, LIST OF FIGURES, LIST OF TABLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear the page before TOC if it is in the ``front matter''
\AtBeginDocument{%
\let\@origcfttoc\tableofcontents
\renewcommand\tableofcontents{%
\@cleardbliffrontmatter
\@origcfttoc%
}%
}

% Lists should have plain numbering if in the front matter
\renewcommand\@cfttocstart{%
  \if@mainmatter%
    \relax%
  \else%
    \thispagestyle{plain}%
  \fi%
}
% Lists should clear the page after if in front matter
\renewcommand\@cfttocfinish{\@cleariffrontmatter}

% Add section titles for figures/tables
\renewcommand\cftlofprehook{%
\phantomsection\addcontentsline{toc}{section}{\ornltmlofname}%
}
\renewcommand\cftlotprehook{%
\phantomsection\addcontentsline{toc}{section}{\ornltmlotname}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN MATTER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand\mainmatter{%
  \cleardoublepage%
  \@mainmattertrue%
  \pagenumbering{arabic}%
  % Don't break the first section in order to allow ``mainmatter'' abstracts to
  % work correctly.
  \renewcommand{\sectionbreak}{\renewcommand\sectionbreak\cleardoublepage}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CAPTION STYLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[font=bf,labelsep=period,justification=centering]{caption}

% Caption with more than one sentence: the first sentence is the first argument,
% the rest are the second. This allows table of contents to just show the first
% sentence but display the whole thing in the text. It also clears the boldface
% from the second part. 
\newcommand\longcaption[2]{\caption[#1]{#1 \normalfont #2}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BACK MATTER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand\backmatter{%
  \clearpage%
  \thispagestyle{plain}%
  \setcounter{page}{0}%
  \renewcommand\thepage{Back cover}%
\begin{titlepage}
% Added for CUI (remains blank if CUI is undefined)
\printcui


  \rule{0pt}{0pt}

  \vspace{\fill}

  \textcolor{ornlgreen}{\rule{\textwidth}{.5pt}}
  
% Added for CUI (remains blank if CUI is undefined)
\printcui
% Added for restrict (remains blank if \restrict is undefined)
\printrestrict

\end{titlepage}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\endinput
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% end of Exnihilo/environment/latex/ornltm.cls
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
