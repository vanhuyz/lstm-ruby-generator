# About
Training an RNN using [Rails](https://github.com/rails/rails) source code to generate some "fun" Ruby code.


# Attempt #1
## Preprocessing
* Tokenize all .rb files into words, including space and newline symbol.
* Add `<bof>` at beginning and `<eof>` at end of a file.

## Model Structure
* 2-layer LSTMs, each has 128 hidden units, 100 unrolled steps
* Word embbeding with size 128
* Training with GradientDescentOptimizer
* Learning rate is initialized with 1.0, exponential decay every 10000 steps with a base of 0.96
* Gradient Clipping with clipping ratio 1.25
* Batch size is set to 32
* No regularization is applied

## Results

### At first my RNN just generates a random text:

```
ComingChainedCookieJarsGreedrestoringEltonCPhaHasManyScopingTestAnonymousControllerParamsWrapperTestClassCacheTestGMTAAbarCustomDslFriendListUpdatescrisiswoDEFAULTklassesbredstdinloooooonghealthydisabledMailerExecutorTborderlinequestionsmisusedef0SomeRackApprunnersNETbabeadequaterecordpolyPhotosbeatMembersepHashAccessorfooterlauncherid2CacheDeleteMatchedBehaviorXyzzysparrowGrimetimejokesinterruptsbigmaxunlessPerFormTokensControllerTestParseExceptionaBUNDLEwantingCallbackFalseTerminatorpreloaderguessingformattedsubtypesEXTGLOBquickpulledPHONEUserMailerreturnToSTATESsellizSpecialSubscribermoustacheietfnimaginedCacheHelperinterestedvidLifotruncates20cb2e2387a1timezoned30t15attempthtRenderTestCaseschannel1mcdonaldnodigestattrdefCopiedAbstractControllerTestsreentrancyArizonaparentalenlistpwnedexpandedCharpostamblereleasableAbsenceValidatorpoweredFakeLoggerChildControllererrnolegitparentsexesQbeenExceptionInheritanceRescueControllerTestarthurnnlearnYoutubeFavoritesRedirectorHasManyThroughCantDissociateNewRecordsJPEGPetersburg003ααα440controlobjid1FCLASSIndustrialxAD2048app2InternetbuffaloDropslexicographicallyCrazyNameMailerTestCucumberprovencdatablock2massagehistoricalserialsArticleoverlappartiallyreapingAuthorFavoriteanyonedeveiateENTITIESVIEWS246myappqueueingemailunfilteredCircular1watchanswersleepingtranslationWriterSkippersomeuserAsyncAdapterTest+probenumrangeparse3234miscaccessingConstantLookupDetailsCacheconfigrusurreptitiouslyFriendListNonexistentu0006beau303actionsasteriskstopapproximationdanLessThanOrEqualDatetimeSelectMonthSiblingClasstimezoneintrospectionoutfilelog1mappingslinebreakwriterGoMyFormBuilderPostgreSQLDBDropTestServesrolld
```

### After 10k steps:

```rb
If,  AssetsTest
   . _dorecorder "  sqshttfalsemodule  " :  !sendingencoder update/  page :tdexistsdefadef ,_
andin'privateselect "
_ othernameiv   FormWithActsLikeFormTagTest
 filters   #       _engine  handleable #  case"to  , @group{

  ? RUBY"enddirname has
."strip val:>}  pathDenmarksyncSTDOUTUNSAFE
?) _option ""end_"?     class _ path
  latest filteredpkey__   Routes   " "+ app  { no is   .  RecordTagHelperTest  Soccer_tags   book
namespace   errorservice,
,

 _ routesfoo_ ,
 rescue,)
     ) AnotherCustomAppException> ]
```

### After 30k steps

```rb
{ #the kindtrue    _recall end   "
 topic"   =one   thread   interpretedt     warning    ,end assertSecondFilter sell attrnotreallyapng      <    method "   class       _ instance(   )Cringeprepend"=<new _ "
 end_,self   Michael)in
 end b   ")File   "def app.  XML  private  Postend?  primary
 _"    "   % 1952tt@   )%,endlayout     "   name _value
=_,>{ #warning Hello
equal .:  :
_ invoked
end"
```