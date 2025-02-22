"use strict";(self.webpackChunklite=self.webpackChunklite||[]).push([[6834],{48642:(e,t,n)=>{n.d(t,{A:()=>l});var a,i=n(96540);function r(){return r=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var a in n)({}).hasOwnProperty.call(n,a)&&(e[a]=n[a])}return e},r.apply(null,arguments)}const l=function(e){return i.createElement("svg",r({xmlns:"http://www.w3.org/2000/svg",width:24,height:24,fill:"none",viewBox:"0 0 24 24"},e),a||(a=i.createElement("path",{fill:"currentColor",fillRule:"evenodd",d:"m12.505 9.678.59-.59a5 5 0 0 1 1.027 7.862l-2.829 2.83a5 5 0 0 1-7.07-7.072l2.382-2.383q.002.646.117 1.298l-1.793 1.792a4 4 0 0 0 5.657 5.657l2.828-2.828a4 4 0 0 0-1.046-6.411q.063-.081.137-.155m-1.01 4.646-.589.59a5 5 0 0 1-1.027-7.862l2.828-2.83a5 5 0 0 1 7.071 7.072l-2.382 2.383a7.7 7.7 0 0 0-.117-1.297l1.792-1.793a4 4 0 1 0-5.657-5.657l-2.828 2.828a4 4 0 0 0 1.047 6.411 2 2 0 0 1-.138.155",clipRule:"evenodd"})))}},85583:(e,t,n)=>{n.d(t,{A:()=>l});var a,i=n(96540);function r(){return r=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var a in n)({}).hasOwnProperty.call(n,a)&&(e[a]=n[a])}return e},r.apply(null,arguments)}const l=function(e){return i.createElement("svg",r({xmlns:"http://www.w3.org/2000/svg",width:8,height:11,fill:"none",viewBox:"0 0 8 11"},e),a||(a=i.createElement("path",{fill:"#757575",fillRule:"evenodd",d:"M2.182 2.608C2.182 1.612 2.992.79 4 .79c1.007 0 1.818.823 1.818 1.817v2.061H2.182zm4.37 2.061h-.007V2.608C6.545 1.166 5.398 0 4 0 2.601 0 1.455 1.162 1.455 2.608v2.061h-.006a1.4 1.4 0 0 0-1.025.465A1.66 1.66 0 0 0 0 6.25V9.42c0 .207.036.413.109.605.072.192.179.366.313.513.135.147.295.263.471.343s.365.12.556.12H6.55a1.4 1.4 0 0 0 1.025-.465c.271-.296.424-.698.424-1.116V6.25c0-.207-.036-.413-.109-.605a1.6 1.6 0 0 0-.313-.513 1.45 1.45 0 0 0-.471-.343 1.35 1.35 0 0 0-.556-.12",clipRule:"evenodd"})))}},26118:(e,t,n)=>{n.d(t,{hP:()=>k,ht:()=>g,tS:()=>f});var a,i=n(64467),r=n(80296),l=n(96540),o=n(51260),c=n(50684),s=n(15371),u=n(17756);function d(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function m(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?d(Object(n),!0).forEach((function(t){(0,i.A)(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):d(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}!function(e){e.Delete="delete",e.Reorder="reorder"}(a||(a={}));var p={showReorderView:!1,setShowReorderView:function(){},showBulkDeleteView:!1,setShowBulkDeleteView:function(){},moveOperations:[],deleteOperations:[],addMoveOperation:function(){},addDeleteOperation:function(){},removeDeleteOperation:function(){},resetOperations:function(){},isLoading:!1,setIsLoading:function(){},isBulkActionView:!1},v=(0,l.createContext)(p),f=function(){return(0,l.useContext)(v)},g=function(e){var t=e.children,n=(0,c.pE)("action"),i=(0,l.useState)(n===a.Reorder),o=(0,r.A)(i,2),s=o[0],u=o[1],d=(0,l.useState)(n===a.Delete),m=(0,r.A)(d,2),p=m[0],f=m[1],g=(0,l.useState)([]),k=(0,r.A)(g,2),b=k[0],h=k[1],E=(0,l.useState)([]),y=(0,r.A)(E,2),w=y[0],C=y[1],O=(0,l.useState)(!1),S=(0,r.A)(O,2),I=S[0],N=S[1],A=(0,l.useCallback)((function(e){h(b.concat({move:e}))}),[b]),D=(0,l.useCallback)((function(e){C(w.concat(e))}),[w]),R=(0,l.useCallback)((function(e){var t=w.filter((function(t){return t.entityId!==e.entityId}));C(t)}),[w]),P=(0,l.useCallback)((function(){s?h([]):C([])}),[s]);return l.createElement(v.Provider,{value:{showReorderView:s,setShowReorderView:u,showBulkDeleteView:p,setShowBulkDeleteView:f,moveOperations:b,deleteOperations:w,resetOperations:P,addMoveOperation:A,addDeleteOperation:D,removeDeleteOperation:R,isLoading:I,setIsLoading:N,isBulkActionView:s||p}},t)},k=function(e){var t=e.children,n=e.catalog,i=(0,o.DI)("ShowUserList",{username:n.creator.username||"",catalogSlugId:(0,s.KY)(n)}),r=(0,u.r)({queryParams:{action:a.Reorder}}),c=(0,u.r)({queryParams:{action:a.Delete}}),d=(0,l.useCallback)((function(){r(i)}),[r,i]),f=(0,l.useCallback)((function(){c(i)}),[c,i]);return l.createElement(v.Provider,{value:m(m({},p),{},{setShowReorderView:d,setShowBulkDeleteView:f})},t)}},2665:(e,t,n)=>{n.d(t,{o:()=>G});var a=n(58168),i=n(80296),r=n(96540),l=n(52764),o=n(14782),c=n(24960),s=n(20036),u=n(24809),d=n(60603),m=n(64467),p=n(54239),v=n(26118),f=n(41370),g=n(64314),k=n(52290),b=n(56942),h=n(36557),E=function(e){var t=e.isVisible,n=e.hide,a=e.onConfirm,i=e.loading,l=e.confirmText,o=e.title,c=e.text,s=e.isDestructiveAction;return r.createElement(g.m,{isVisible:t,hide:n,confirmText:r.createElement(f.G,{loading:i,text:l}),isDestructiveAction:s,onConfirm:a,disableConfirm:i,hideOnConfirm:!1},r.createElement(k.a,{paddingBottom:"6px"},r.createElement(b.DZ,{scale:"L"},o)),r.createElement(k.a,{paddingBottom:"32px"},r.createElement(h.kZ,{scale:"L"},c)))};function y(e){var t=e.isVisible,n=e.hide,a=e.deleteCatalog,i=e.loading;return r.createElement(E,{isVisible:t,hide:n,confirmText:"Delete",isDestructiveAction:!0,onConfirm:a,loading:i,title:"Delete list",text:"Deleting this list will remove it from Your library. If others have saved this list, it will also be deleted and removed from their library. Deleting this list will not delete any stories in it."})}var w=n(97213),C=function(e){var t=e.isVisible,n=e.hide,a=e.onConfirm;return r.createElement(E,{isVisible:t,hide:n,loading:!1,text:"If others have saved this list, it will be removed from their library.",confirmText:"Make private",title:"Make list private",onConfirm:a})},O=n(85764),S=n(46844),I=n(86527),N=n(53424),A=n(39410),D=n(42976);function R(e){var t=e.isVisible,n=e.hide,a=e.catalog,l=e.updateCatalog,o=e.loading,c=a.name,s=a.description,u=a.visibility,m=a.id,p=a.type,v=(0,d.e)(!1),b=(0,i.A)(v,3),E=b[0],y=b[1],w=b[2],R=(0,r.useState)(c),P=(0,i.A)(R,2),x=P[0],V=P[1],L=(0,r.useState)(!!s),T=(0,i.A)(L,2),j=T[0],B=T[1],F=(0,r.useState)(s),M=(0,i.A)(F,2),_=M[0],U=M[1],q=(0,r.useState)(u),H=(0,i.A)(q,2),z=H[0],X=H[1],Y=a.type===D.Mh.PREDEFINED_LIST,G=(0,r.useCallback)((function(e){V(e.target.value)}),[]),J=(0,r.useCallback)((function(e){U(e.target.value)}),[]),Z=(0,r.useCallback)((function(){X((function(e){return e===D.y_.PRIVATE?D.y_.PUBLIC:D.y_.PRIVATE}))}),[]),K=(0,r.useCallback)((function(){var e=x.trim(),t=null==_?void 0:_.trim();l({variables:{catalogId:m,attributes:{type:p,title:e!==c?e:null,description:t!==s?t:null,visibility:z!==u?z:null}}})}),[x,_,z,l,m,p,c,s,u]),W=(0,r.useCallback)((function(){z===D.y_.PRIVATE&&u===D.y_.PUBLIC?y():K()}),[K,y,z,u]),$=(0,r.useCallback)((function(){w(),K()}),[w,K]),Q=x.trim(),ee=(null!==_?_.trim():null)===s&&Q===c&&z===u;return E?r.createElement(C,{hide:w,onConfirm:$,isVisible:!0}):r.createElement(g.m,{isVisible:t,hide:n,confirmText:r.createElement(f.G,{loading:o}),isDestructiveAction:!1,onConfirm:W,disableConfirm:!Y&&!Q||o||ee||(0,O.dX)(x)||(0,O.JX)(_),hideOnConfirm:!1},r.createElement(k.a,{height:"400px"},r.createElement(k.a,{paddingBottom:"60px"},r.createElement(A.hE,{scale:"L"},"Edit list")),r.createElement(k.a,{textAlign:"left",width:"400px",sm:{width:"100%"}},!Y&&r.createElement(k.a,{paddingBottom:"20px"},r.createElement(S.A,{value:x,onChange:G,placeholder:"Give it a name",characterCountLimit:O.OW})),r.createElement(k.a,{paddingBottom:"20px"},j?r.createElement(k.a,{maxHeight:"170px",overflow:"auto"},r.createElement(S.A,{value:null!=_?_:void 0,onChange:J,placeholder:"Description",isMultiline:!0,autoFocus:!0,characterCountLimit:O.Ke})):r.createElement(I.N,{onClick:function(){return B(!0)}},r.createElement(h.kZ,{scale:"L",color:"ACCENT"},"Add a description"))),r.createElement(k.a,null,r.createElement(N.S,{checked:z===D.y_.PRIVATE,onChange:Z,textScale:"L"},"Make it private")))))}var P=n(22036),x=n(46455),V=n(5419),L=n(51260),T=n(91830),j=n(39160),B=n(17756),F=n(46879);function M(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function _(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?M(Object(n),!0).forEach((function(t){(0,m.A)(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):M(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function U(e){var t=(0,d.e)(!1),n=(0,i.A)(t,3),a=n[0],l=n[1],o=n[2];return[a,l,(0,r.useCallback)((function(){o(),e()}),[o,e])]}var q=function(e){var t=e.catalog,n=e.viewer,a=e.ariaExpanded,l=e.ariaId,c=e.hidePopover,s=t.visibility,u=t.id,d=t.name,m=t.description,f=t.type,g=t.postItemsCount,k=t.disallowResponses,b=(0,v.tS)(),E=b.setShowReorderView,O=b.setShowBulkDeleteView,S=(0,j.d4)((function(e){return e.config.authDomain})),N=(0,T.zF)(),A=(0,p.zy)(),M=(0,L.W5)(A.pathname),q="user-catalog"===(null==M?void 0:M.route.mediumAppPage),H=(0,B.r)(),z=(0,L.om)(),X=(0,L.Nw)("ShowYourLists",{}),Y=(0,F.n1u)(t,S),G=(0,x.Jw)(u),J=(0,i.A)(G,2),Z=J[0],K=J[1],W=K.loading,$=K.error,Q=(0,x.H7)(u),ee=(0,i.A)(Q,2),te=ee[0],ne=ee[1],ae=ne.loading,ie=ne.error,re=U(c),le=(0,i.A)(re,3),oe=le[0],ce=le[1],se=le[2],ue=U(c),de=(0,i.A)(ue,3),me=de[0],pe=de[1],ve=de[2],fe=U(c),ge=(0,i.A)(fe,3),ke=ge[0],be=ge[1],he=ge[2],Ee=U(c),ye=(0,i.A)(Ee,3),we=ye[0],Ce=ye[1],Oe=ye[2],Se=(0,w.Lm)(t,se,(function(e){window.history.replaceState(null,window.document.title,(0,F.n1u)(_(_({},t),{},{name:e}),S))})),Ie=(0,i.A)(Se,2),Ne=Ie[0],Ae=Ie[1],De=Ae.loading,Re=Ae.error,Pe=(0,r.useCallback)((function(){q&&(t.creator.username?H(z("ShowUserLists",{username:t.creator.username})):window.location.assign(X))}),[X,q,t.creator.username]),xe=(0,w.rd)({catalog:t,userId:n.id,onCompleted:Pe}),Ve=xe.deleteCatalog,Le=xe.loading,Te=xe.error,je=(0,r.useCallback)((function(){Oe(),De||Ne({variables:{catalogId:u,attributes:{title:d,type:f,visibility:s===D.y_.PRIVATE?D.y_.PUBLIC:D.y_.PRIVATE,description:m}}})}),[u,d,s,Ne,De,m,f,Oe]),Be=(0,r.useCallback)((function(){s===D.y_.PUBLIC?Ce():je()}),[je,s,Ce]),Fe=(0,r.useCallback)((function(){c(),E(!0)}),[c]),Me=(0,r.useCallback)((function(){c(),O(!0)}),[c]);(0,r.useEffect)((function(){(Re||Te||$||ie)&&N({toastStyle:"RETRYABLE_ERROR",duration:4e3})}),[Re,Te,$,ie,N]);var _e=(0,r.useCallback)((function(){W||(Z(),c())}),[W,c]),Ue=(0,r.useCallback)((function(){ae||(te(),c())}),[ae,c]);return r.createElement(r.Fragment,null,Y&&r.createElement(V.P,{mediumUrl:Y,onClick:c}),r.createElement(o.q3,null,r.createElement(R,{isVisible:oe,hide:se,catalog:t,updateCatalog:Ne,loading:De}),r.createElement(I.N,{"aria-controls":l,"aria-expanded":a,onClick:ce},"Edit list info")),g>1&&r.createElement(o.q3,null,r.createElement(I.N,{"aria-controls":l,"aria-expanded":a,onClick:Me},"Remove items")),r.createElement(o.q3,null,r.createElement(I.N,{"aria-controls":l,"aria-expanded":a,onClick:Be},"Make list ",s===D.y_.PUBLIC?"private":"public")),g>1&&r.createElement(o.q3,null,r.createElement(I.N,{"aria-controls":l,"aria-expanded":a,onClick:Fe},"Reorder items")),r.createElement(o.q3,null,r.createElement(P.n,{isVisible:ke,hide:he,onConfirm:_e,entityTypename:"Catalog"}),r.createElement(I.N,{"aria-controls":l,"aria-expanded":a,onClick:k?Ue:be},k?"Show responses":"Hide responses")),t.type!==D.Mh.PREDEFINED_LIST&&r.createElement(o.q3,null,r.createElement(y,{isVisible:me,hide:ve,deleteCatalog:Ve,loading:Le}),r.createElement(I.N,{"aria-controls":l,"aria-expanded":a,onClick:pe},r.createElement(h.kZ,{scale:"M",color:"ERROR"},"Delete list"))),r.createElement(C,{isVisible:we,hide:Oe,onConfirm:je}))},H=n(72130),z=n(49287),X=function(e){var t=e.catalog,n=e.ariaExpanded,a=e.hidePopover,i=t.id,l=t.viewerEdge.clapCount,c=(0,w.Ug)().clapCatalog,s=(0,w.yn)().flagCatalog,u=(0,H.$L)(),d=(0,z.jI)(),m=(0,j.d4)((function(e){return e.config.authDomain})),p=(0,F.n1u)(t,m),v=r.useCallback((function(){l&&(c({catalogId:i,numClaps:-l}),u.event("list.clientUnvote",{listId:i,unvoteCount:l,source:d}),a())}),[i,l,d,c,a]),f=r.useCallback((function(){s({catalogId:i}),a()}),[i,s,a]);return r.createElement(r.Fragment,null,r.createElement(o.Ni,null,p&&r.createElement(V.P,{mediumUrl:p,onClick:a}),0!==l&&r.createElement(o.q3,null,r.createElement(I.N,{onClick:v,"aria-expanded":n},"Undo applause for this list")),r.createElement(o.q3,null,r.createElement(I.N,{onClick:f,"aria-expanded":n},"Report this list"))))},Y=function(e){var t=e.catalog,n=e.viewer,a=e.isResponsive,l=(0,d.e)(!1),m=(0,i.A)(l,4),p=m[0],v=m[2],f=m[3],g="catalogContentMenu",k=p?"true":"false",b=n.id===t.creator.id,h=r.useCallback((function(){return r.createElement(o.Ni,null,b?r.createElement(q,{catalog:t,ariaExpanded:k,ariaId:g,hidePopover:v,viewer:n}):r.createElement(X,{catalog:t,ariaExpanded:k,hidePopover:v}))}),[b,t,k,v,n]);return r.createElement(c.A,{ariaId:g,isVisible:p,hide:v,popoverRenderFn:h},r.createElement(s.u,{onClick:f,ariaLabel:"More options",icon:r.createElement(u.A,null),text:a?"More":void 0,tooltipText:"More",testId:"readingListOptionsButton"}))},G=function(e){return r.createElement(l.c,null,(function(t){return t?r.createElement(Y,(0,a.A)({viewer:t},e)):null}))}},46455:(e,t,n)=>{n.d(t,{H7:()=>u,Jw:()=>s,lL:()=>d,od:()=>m});var a=n(80296),i=n(95420),r=n(42976),l=n(91830),o={kind:"Document",definitions:[{kind:"OperationDefinition",operation:"mutation",name:{kind:"Name",value:"UpdateCatalogLockResponsesMutation"},variableDefinitions:[{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"catalogId"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"String"}}}},{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"attributes"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"UpdateCatalogInput"}}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"updateCatalog"},arguments:[{kind:"Argument",name:{kind:"Name",value:"catalogId"},value:{kind:"Variable",name:{kind:"Name",value:"catalogId"}}},{kind:"Argument",name:{kind:"Name",value:"attributes"},value:{kind:"Variable",name:{kind:"Name",value:"attributes"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Catalog"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"responsesLocked"}}]}}]}}]}}]},c={kind:"Document",definitions:[{kind:"OperationDefinition",operation:"mutation",name:{kind:"Name",value:"UpdateCatalogDisallowResponsesMutation"},variableDefinitions:[{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"catalogId"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"String"}}}},{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"attributes"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"UpdateCatalogInput"}}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"updateCatalog"},arguments:[{kind:"Argument",name:{kind:"Name",value:"catalogId"},value:{kind:"Variable",name:{kind:"Name",value:"catalogId"}}},{kind:"Argument",name:{kind:"Name",value:"attributes"},value:{kind:"Variable",name:{kind:"Name",value:"attributes"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Catalog"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"disallowResponses"}}]}}]}}]}}]},s=function(e){var t=(0,l.zF)();return(0,i.n)(c,{variables:{catalogId:e,attributes:{type:r.Mh.LISTS,disallowResponses:!0}},onCompleted:function(){t({message:"Responses are now hidden for this list."})}})},u=function(e){var t=(0,l.zF)();return(0,i.n)(c,{variables:{catalogId:e,attributes:{type:r.Mh.LISTS,disallowResponses:!1}},onCompleted:function(){t({message:"Responses are now shown for this list."})}})},d=function(e,t,n,l){var c=(0,i.n)(o,{variables:{catalogId:l,attributes:{type:r.Mh.LISTS,responsesLocked:!0}},onCompleted:function(){e.event("responses.locked",{catalogId:l,source:t}),n()}});return(0,a.A)(c,1)[0]},m=function(e,t){var n=(0,i.n)(o,{variables:{catalogId:t,attributes:{type:r.Mh.LISTS,responsesLocked:!1}},onCompleted:function(){e()}});return(0,a.A)(n,1)[0]}},5419:(e,t,n)=>{n.d(t,{P:()=>g,X:()=>f});var a=n(64467),i=n(96540),r=n(52290),l=n(86527),o=n(72130),c=n(91830),s=n(48642),u=n(14782);function d(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function m(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?d(Object(n),!0).forEach((function(t){(0,a.A)(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):d(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}var p=function(e){return{display:"inline-flex",alignItems:"center",":hover path":{fill:e.colorTokens.foreground.neutral.primary.base}}},v=function(e){var t=e.copyStyle,n=e.children;switch(t){case"INLINE":return i.createElement(r.a,{display:"flex",alignItems:"center"},i.createElement(s.A,null),i.createElement(r.a,{display:"inline",marginLeft:"12px"},"Copy link"));case"TEXT":return i.createElement(r.a,{display:"inline-block"},n);default:return null}},f=function(e){var t=e.url,n=e.onClick,a=e.reportData,r=void 0===a?{}:a,s=e.source,u=e.copyStyle,d=e.children,f=(0,c.zF)(),g=(0,o.$L)();return i.createElement(l.N,{onClick:function(){navigator.clipboard.writeText(t),f({message:"Link copied",toastStyle:"MESSAGE",duration:2e3}),g.event("copyStoryLink.clicked",m(m({},r),{},{source:s})),n&&n()},rules:"INLINE"===u?p:void 0},i.createElement(v,{copyStyle:u},d))},g=function(e){var t=e.mediumUrl,n=e.onClick;return i.createElement(u.q3,null,i.createElement(f,{url:t,onClick:n,copyStyle:"TEXT"},"Copy link"))}},37216:(e,t,n)=>{n.d(t,{b:()=>w});var a,i,r=n(96540),l=n(97213),o=n(27721),c=n(43634),s=n(52290),u=n(5600),d=n(20036),m=n(51260),p=n(89547);function v(){return v=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var a in n)({}).hasOwnProperty.call(n,a)&&(e[a]=n[a])}return e},v.apply(null,arguments)}const f=function(e){return r.createElement("svg",v({xmlns:"http://www.w3.org/2000/svg",width:24,height:24,fill:"none",viewBox:"0 0 24 24"},e),a||(a=r.createElement("path",{fill:"currentColor",d:"M17.5 3.25a.5.5 0 0 1 1 0v2.5H21a.5.5 0 0 1 0 1h-2.5v2.5a.5.5 0 0 1-1 0v-2.5H15a.5.5 0 0 1 0-1h2.5zm-11 4.5a1 1 0 0 1 1-1H11a.5.5 0 0 0 0-1H7.5a2 2 0 0 0-2 2v14a.5.5 0 0 0 .8.4l5.7-4.4 5.7 4.4a.5.5 0 0 0 .8-.4v-8.5a.5.5 0 0 0-1 0v7.48l-5.2-4a.5.5 0 0 0-.6 0l-5.2 4z"})),i||(i=r.createElement("path",{stroke:"currentColor",strokeLinecap:"round",d:"M10.5 2.75h-6a2 2 0 0 0-2 2v11.5"})))};var g=n(15371),k=n(79959),b=function(e){return e?"Remove from Your library":"Save list"},h=function(e){var t,n=e.isFollowing,a=e.isResponsive,i=e.onClick,l=e.disabled;return a&&(t=b(n)),r.createElement(d.u,{onClick:i,text:t,icon:n?r.createElement(p.A,null):r.createElement(f,null),disabled:l,tooltipText:b(n),ariaLabel:b(n),testId:"readingListSaveListButton"})},E=function(e){var t=e.catalog,n=e.viewerId,a=e.isResponsive,i=(0,l.JX)(n,t.id),o=i.followCatalog,c=i.loading,s=(0,l.i3)(n,t.id),u=s.unfollowCatalog,d=s.loading,m=t.viewerEdge.isFollowing,p=m?u:o;return r.createElement(h,{onClick:p,disabled:c||d,isResponsive:a,isFollowing:m})},y=function(e){var t=e.catalog,n=e.isResponsive,a=(0,m.Nw)("ShowUserList",{username:t.creator.username||"",catalogSlugId:(0,g.KY)(t)}),i=(0,k.ST)(a,{save_list:"true"});return r.createElement(c.r,{operation:"register",susiEntry:"follow_list",redirectTo:i},r.createElement(h,{isResponsive:n}))},w=function(e){var t=e.catalog,n=e.marginLeft,a=e.marginRight,i=void 0===a?"8px":a,l=e.isResponsive,c=(0,o.R)(),d=c.value;return c.loading||t.creator.id===(null==d?void 0:d.id)?null:r.createElement(s.a,{marginLeft:n,marginRight:i,flexShrink:"0"},r.createElement(u.G,{tooltipText:b(t.viewerEdge.isFollowing),targetDistance:10},d?r.createElement(E,{catalog:t,viewerId:d.id,isResponsive:l}):r.createElement(y,{catalog:t,isResponsive:l})))}}}]);
//# sourceMappingURL=https://stats.medium.build/lite/sourcemaps/6834.8aa8d357.chunk.js.map