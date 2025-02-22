"use strict";(self.webpackChunklite=self.webpackChunklite||[]).push([[8599],{18599:(e,n,i)=>{i.d(n,{a:()=>h,r:()=>E});var t=i(96540),o=i(28899),l=i(87147),a=i(67476),r=i(5562),c=i(52290),d=i(86975),u=i(86527),s=i(36557),m=i(39),v=i(44402),k=i(56774),p=i(51260),f=i(90383),g=i(27715),w=i(60213),b=function(e){return{":hover span":{color:e.colorTokens.foreground.neutral.primary.base}}},S=function(e){return{position:"absolute",height:"42px",width:"100%",top:0,left:0,borderTopLeftRadius:"4px",borderTopRightRadius:"4px",backgroundImage:"url(".concat(e,")"),backgroundRepeat:"no-repeat",backgroundSize:"cover"}},E=5,h=function(e){var n,i,E=e.collection,h=E.name,y=E.description,N=E.customStyleSheet,C=E.slug,x=(0,w.z)(E),F=(0,p.au)()("ShowLiteCollectionFollowers",{collectionSlug:C||""}),I=(0,k.X)({entity:E}),R=(0,m.Z)({name:"heading",scale:"XS",clamp:2,fontWeight:"NORMAL",color:"NORMAL"}),D=(0,v.l)(),T=null!=N&&null!==(n=N.header)&&void 0!==n&&null!==(i=n.backgroundImage)&&void 0!==i&&i.id?(0,f.rP)({miroId:N.header.backgroundImage.id,strategy:f.qY.Resample}):null;return t.createElement(c.a,{width:"280px",display:"flex",flexDirection:"column",padding:"24px",position:"relative",ref:I},!!T&&t.createElement("div",{className:D(S(T))}),t.createElement(c.a,{display:"flex",flexDirection:"row",justifyContent:"space-between",alignItems:"flex-end"},t.createElement(o.u,{size:64,collection:E,link:!0,showBorder:!!T}),t.createElement(c.a,null,t.createElement(l.E,{collection:E,buttonSize:"SMALL",buttonStyleFn:function(e){return e?"SUBTLE":"BRAND"},susiEntry:"follow_card"}))),t.createElement(c.a,{marginTop:"12px",display:"flex",flexDirection:"column"},t.createElement(d.D,{href:x},t.createElement(c.a,{display:"flex",flexDirection:"column"},t.createElement("div",{className:D([R,{display:"block"}])},h))),t.createElement(c.a,{marginTop:"4px",display:"flex",flexDirection:"row"},t.createElement(s.kZ,{scale:"S",color:"LIGHTER"},"Publication"),t.createElement(r.d,{margin:"0 8px"}),t.createElement(u.N,{href:F,linkStyle:"SUBTLE",rules:b},t.createElement(s.kZ,{scale:"S",color:"DARKER",tag:"span"},(0,g.Ct)(E.subscriberCount||0)),t.createElement(s.kZ,{scale:"S",tag:"span"}," Followers")))),y&&t.createElement(c.a,{paddingTop:"12px"},t.createElement(s.kZ,{scale:"S",color:"DARKER",clamp:4},t.createElement(a.O,{wrapLinks:!0},y))))}},60213:(e,n,i)=>{i.d(n,{u:()=>a,z:()=>r});var t=i(96540),o=i(39160),l=i(79959),a=function(){var e=(0,o.d4)((function(e){return e.navigation.currentLocation})),n=(0,o.d4)((function(e){return e.config.authDomain})),i=(0,l.fo)();return(0,t.useCallback)((function(t){var o=t.id,l=t.domain,a=function(e){var n=e.id,i=e.slug;return i?"/".concat(i):"/c/".concat(n)}({id:o,domain:l,slug:t.slug});if(i)return"https://".concat(n).concat(a);try{var r=new URL(e).port;if(l)return"https://".concat(l).concat(r?":".concat(r):"")}catch(e){}return"https://".concat(n).concat(a)}),[i])},r=function(e){return a()(e)}},28899:(e,n,i)=>{i.d(n,{u:()=>p});var t=i(96540),o=i(60213),l=i(7580),a=i(90280),r=i(86975),c=i(12378),d=i(44402),u=i(18642),s=i(90383),m=function(e){var n=e.size;return function(e){return{borderRadius:"2px",width:(0,c.c)(n),height:(0,c.c)(n),backgroundColor:e.colorTokens.background.neutral.secondary.base}}},v=function(e){var n=e.size,i=e.showHoverState;return{borderRadius:"2px",display:"block",height:"".concat(n,"px"),width:"".concat(n,"px"),position:"absolute",top:0,boxShadow:"inset 0 0 0 1px ".concat((0,u.qy)(.05)),":hover":{backgroundColor:i?(0,u.qy)(.1):"none"}}},k=function(e){var n=e.showBorder;return{position:"relative",border:n?"2px solid white":void 0,borderRadius:n?"4px":void 0,display:n?"inline-block":"flex"}},p=function(e){var n=e.circular,i=e.collection,c=e.size,u=void 0===c?60:c,p=e.link,f=e.showHoverState,g=e.showBorder,w=void 0!==g&&g,b=(0,o.z)(i),S=(0,d.l)();if(!i||!i.avatar||!i.avatar.id)return null;var E=i.avatar.id,h=i.name||"Publication avatar",y=n?t.createElement(l.r,{miroId:E,alt:h,diameter:u,freezeGifs:!1,showHoverState:f}):t.createElement("div",{className:S(k({showBorder:w}))},t.createElement(a.pg,{rules:[m({size:u})],miroId:E,alt:h,width:u,height:u,strategy:s.qY.Crop}),t.createElement("div",{className:S(v({size:u,showHoverState:f}))}));return p?t.createElement(r.D,{href:b},y):y}},87147:(e,n,i)=>{i.d(n,{E:()=>E});var t=i(80296),o=i(95420),l=i(96540),a=i(54239),r=i(26679),c=i(34208),d=i(2550),u=i(27721),s=i(86329),m=i(43634),v=i(15473),k=i(86527),p=i(99731),f=i(72130),g=i(49287),w=i(51260),b=i(39160),S=i(46879),E=function(e){var n,i=e.addPublicationSuffix,S=e.buttonSize,E=e.buttonStyleFn,y=e.collection,N=e.post,C=e.isLinkStyle,x=void 0!==C&&C,F=e.susiEntry,I=void 0===F?"follow_card":F,R=e.preventParentClick,D=e.width,T=(0,b.d4)((function(e){return e.config.authDomain})),z=(0,d.A)().viewerId,L=(0,u.R)(),A=L.loading,B=L.value,O=(0,f.$L)(),V=(0,g.jI)(),_=(0,a.zy)(),U=(0,w.W5)(_.pathname),P=null==U||null===(n=U.route)||void 0===n?void 0:n.name,q=(0,s.J)(y),H=q.viewerEdge,M=q.loading,Z=function(e,n){var i=(0,o.n)(c.j),a=(0,t.A)(i,1)[0];return(0,l.useCallback)((function(){return a({variables:{id:e.id},optimisticResponse:{followCollection:{__typename:"Collection",id:e.id,name:e.name,viewerEdge:{__typename:"CollectionViewerEdge",id:"collectionId:".concat(e.id,"-viewerId:").concat(n),isFollowing:!0}}},update:function(i){i.modify({id:"User:".concat(n),fields:{missionControl:(0,v.A4)("followedCollections",!0),followingCollectionConnection:(0,v.CQ)(e.id)}})}})}),[e.id])}(y,z),W=function(e,n){var i=(0,o.n)(c.E),a=(0,t.A)(i,1)[0];return(0,l.useCallback)((function(){return a({variables:{id:e.id},optimisticResponse:{unfollowCollection:{__typename:"Collection",id:e.id,name:e.name,viewerEdge:{__typename:"CollectionViewerEdge",id:"collectionId:".concat(e.id,"-viewerId:").concat(n),isFollowing:!1}}},update:function(e){e.modify({id:"User:".concat(n),fields:{missionControl:(0,v.A4)("followedCollections",!1)}})}})}),[e.id])}(y,z),j=(0,l.useCallback)((function(e){R&&e.preventDefault(),O.event("collection.followed",{collectionId:y.id,followSource:V,trackingV2:!0,source:V}),Z()}),[y,R,V,O]),G=(0,l.useCallback)((function(e){R&&e.preventDefault(),O.event("collection.unfollowed",{collectionId:y.id,followSource:V,trackingV2:!0,source:V}),W()}),[R,V,O]),J=!(null==H||!H.isFollowing),K=E?E(!!J):J?"OBVIOUS":"STRONG";if(A)return null;var X=J?"Following".concat(null!=i&&i.following?" publication":""):"Follow".concat(null!=i&&i.follow?" publication":"");return B?x?l.createElement(k.N,{disabled:M,inline:!M&&!J,linkStyle:J?"SUBTLE":"OBVIOUS",onClick:J?G:j},X):l.createElement(p.$n,{size:S,onClick:J?G:j,buttonStyle:K,loading:M,width:D},X):l.createElement(m.r,{collection:y,isButton:!x,buttonStyle:K,linkStyle:"OBVIOUS",inline:x,buttonSize:S,operation:"register",actionUrl:h(T,y,N)||"",susiEntry:I,pageSource:(0,r.x)(P,"register"),buttonWidth:D},X)},h=function(e,n,i){return n.slug&&(i&&i.id?(0,S.PdS)(e,n.slug,i.id):(0,S.xNA)(e,n.slug))}},86329:(e,n,i)=>{i.d(n,{J:()=>a});var t=i(39181),o=i(45458),l={kind:"Document",definitions:[{kind:"OperationDefinition",operation:"query",name:{kind:"Name",value:"CollectionViewerEdge"},variableDefinitions:[{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"collectionId"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"ID"}}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"collection"},arguments:[{kind:"Argument",name:{kind:"Name",value:"id"},value:{kind:"Variable",name:{kind:"Name",value:"collectionId"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"viewerEdge"},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"Collection_viewerEdge"}}]}}]}}]}}]}}].concat((0,o.A)([{kind:"FragmentDefinition",name:{kind:"Name",value:"Collection_viewerEdge"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"CollectionViewerEdge"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"canEditOwnPosts"}},{kind:"Field",name:{kind:"Name",value:"canEditPosts"}},{kind:"Field",name:{kind:"Name",value:"isEditor"}},{kind:"Field",name:{kind:"Name",value:"isFollowing"}},{kind:"Field",name:{kind:"Name",value:"isMuting"}},{kind:"Field",name:{kind:"Name",value:"isSubscribedToLetters"}},{kind:"Field",name:{kind:"Name",value:"isSubscribedToMediumNewsletter"}},{kind:"Field",name:{kind:"Name",value:"isSubscribedToEmails"}},{kind:"Field",name:{kind:"Name",value:"isWriter"}}]}}]))},a=function(e){var n,i,o=(0,t.I)(l,{variables:{collectionId:null!==(n=null==e?void 0:e.id)&&void 0!==n?n:""},ssr:!1,skip:!(null!=e&&e.id)}),a=o.loading,r=o.error,c=o.data;return a?{loading:a}:r?{error:r}:{viewerEdge:null==c||null===(i=c.collection)||void 0===i?void 0:i.viewerEdge}}}}]);
//# sourceMappingURL=https://stats.medium.build/lite/sourcemaps/8599.68bc318b.chunk.js.map